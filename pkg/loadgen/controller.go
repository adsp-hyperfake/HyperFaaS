package loadgen

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"sync"
	"time"

	"gopkg.in/yaml.v2"
)

type Controller struct {
	Config            *Config
	collector         *Collector
	funcMgr           *FunctionManager
	funcDataProviders map[string]DataProvider
	l                 *slog.Logger
}

type Config struct {
	GenerateWorkload   bool                          `yaml:"generate_workload"`
	LeafAddress        string                        `yaml:"leaf_address"`
	Seed               int64                         `yaml:"seed,omitempty"`
	MaxDuration        time.Duration                 `yaml:"max_duration"`
	Timeout            int32                         `yaml:"timeout"`
	Patterns           map[string]*PhasePattern      `yaml:"patterns"`
	Workload           *Workload                     `yaml:"workload,omitempty"`
	FunctionConfig     map[string]*FunctionConfig    `yaml:"function_config"`
	DataProviders      map[string]*DataProviderConfig `yaml:"data_providers,omitempty"`
}


type Workload struct {
	LeafAddress string        `yaml:"leaf_address"`
	MaxDuration time.Duration `yaml:"max_duration"`
	Timeout     int32         `yaml:"timeout"`
	Phases      []TestPhase   `yaml:"phases"` // should be ordered by start time ascending
}

type TestPhase struct {
	Name       string        `yaml:"name"`
	Type       string        `yaml:"type"`       // "constant" | "variable"
	StartTime  time.Duration `yaml:"start_time"` // Relative to workload start
	Duration   time.Duration `yaml:"duration"`
	StartRPS   int           `yaml:"start_rps"`
	EndRPS     int           `yaml:"end_rps,omitempty"`
	Step       int           `yaml:"step,omitempty"` // For ramping increment/decrement
	ImageTag   string        `yaml:"image_tag"`
	FunctionID string        // function target
}

func (c *Controller) Run() {

	client := NewLeafClient(c.Config.LeafAddress)
	fmt.Println("Creating functions")
	c.CreateFunctions()

	fmt.Println("Starting workload, max duration:", c.Config.MaxDuration)
	ctx, cancel := context.WithTimeout(context.Background(), c.Config.MaxDuration)
	defer cancel()

	startTime := time.Now()

	wg := sync.WaitGroup{}

	for _, phase := range c.Config.Workload.Phases {
		wg.Add(1)
		go func(phase TestPhase) {
			defer wg.Done()

			// wait for phase start time
			<-time.After(phase.StartTime)
			c.l.Info("Starting phase", "Phase", phase.Name, "Type", phase.Type, "Start RPS", phase.StartRPS, "End RPS", phase.EndRPS, "Step", phase.Step, "Duration", phase.Duration)

			switch phase.Type {
			case "constant":
				executor := NewConstantExecutor(client, c.collector, c.funcMgr, c.l, c.GetDataProvider(phase.ImageTag))
				executor.Execute(ctx, phase)
			case "variable":
				executor := NewRampingExecutor(client, c.collector, c.funcMgr, c.l, c.GetDataProvider(phase.ImageTag))
				executor.Execute(ctx, phase)
			}

		}(phase)
	}

	wg.Wait()

	log.Println("Workload completed in", time.Since(startTime))
	c.collector.Close()
}

func (c *Controller) CreateFunctions() {
	functions := make(map[string]string)

	for _, phase := range c.Config.Workload.Phases {
		functions[phase.ImageTag] = phase.ImageTag
	}

	for _, imageTag := range functions {
		f := c.funcMgr.CreateFunction(imageTag, c.Config.Workload.Timeout, c.Config.FunctionConfig[imageTag])
		for i, phase := range c.Config.Workload.Phases {
			if phase.ImageTag == imageTag {
				phase.FunctionID = f.ID
				c.Config.Workload.Phases[i] = phase
			}
		}
	}
}

func (c *Controller) GetDataProvider(imageTag string) DataProvider {
	return c.funcDataProviders[imageTag]
}

type Option func(*Controller)

func NewController(logger *slog.Logger, opts ...Option) *Controller {
	c := &Controller{
		l: logger,
	}
	for _, opt := range opts {
		opt(c)
	}

	if c.Config.GenerateWorkload {
		generator := NewWorkloadGenerator(c.Config.Seed, c.Config.MaxDuration, c.Config.LeafAddress, c.Config.Timeout, c.Config.Patterns)
		c.Config.Workload = generator.GenerateWorkload()
	}
	c.funcMgr = NewFunctionManager(c.Config.LeafAddress)
	c.funcDataProviders = make(map[string]DataProvider)

	distinctImageTags := getDistinctImageTags(c.Config.Workload.Phases)

	for _, imageTag := range distinctImageTags {
		config, exists := c.Config.DataProviders[imageTag]
		if !exists {
			c.l.Error("No data provider configuration found for image tag", "imageTag", imageTag)
			continue
		}
		
		provider, err := CreateDataProvider(imageTag, config)
		if err != nil {
			c.l.Error("Failed to create data provider", "imageTag", imageTag, "error", err)
			continue
		}
		c.funcDataProviders[imageTag] = provider
	}

	return c
}

func WithConfigFile(path string) Option {
	return func(c *Controller) {
		c.Config = &Config{}
		f, err := os.ReadFile(path)
		if err != nil {
			log.Fatal("Failed to read config file:", err)
		}
		err = yaml.Unmarshal(f, c.Config)
		if err != nil {
			log.Fatal("Failed to parse config file:", err)
		}

		if c.Config.LeafAddress == "" {
			log.Fatal("Leaf address is required")
		}

		if c.Config.GenerateWorkload && c.Config.Patterns == nil {
			log.Fatal("Generate workload is true, but no patterns are provided")
		}

		if c.Config.MaxDuration == 0 {
			log.Fatal("Max duration is required")
		}

		if c.Config.Timeout == 0 {
			log.Fatal("Timeout is required")
		}

		if c.Config.Workload != nil {
			for _, phase := range c.Config.Workload.Phases {
				if phase.Type != "constant" && phase.Type != "variable" {
					log.Fatal("Phase type must be either constant or variable")
				}
				if phase.Type == "variable" && (phase.EndRPS == 0 || phase.Step == 0) {
					log.Fatal("Step and end RPS are required for variable phases")
				}
				if phase.Type == "constant" && (phase.StartRPS == 0 || phase.EndRPS != 0 || phase.Step != 0) {
					log.Fatal("Start RPS is required for constant phases")
				}
			}
		}
	}
}

func WithCollector(collector *Collector) Option {
	return func(c *Controller) {
		c.collector = collector
	}
}


func getDistinctImageTags(phases []TestPhase) []string {
	imageTags := make(map[string]bool)
	for _, phase := range phases {
		imageTags[phase.ImageTag] = true
	}
	keys := make([]string, 0, len(imageTags))
	for k := range imageTags {
		keys = append(keys, k)
	}
	return keys
}
