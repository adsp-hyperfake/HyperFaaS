package main

import (
	"flag"
	"log"
	"log/slog"
	"os"
	"time"

	kv "github.com/3s-rg-codes/HyperFaaS/pkg/keyValueStore"
	cr "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime"
	dockerRuntime "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime/docker"
	fakeRuntime "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime/fake"
	"github.com/3s-rg-codes/HyperFaaS/pkg/worker/network"
	fakeNetwork "github.com/3s-rg-codes/HyperFaaS/pkg/worker/network/fake"
	"github.com/3s-rg-codes/HyperFaaS/pkg/worker/stats"

	"net/http"
	_ "net/http/pprof"

	"github.com/3s-rg-codes/HyperFaaS/pkg/worker/controller"
)

type WorkerConfig struct {
	General struct {
		Address             string `env:"WORKER_ADDRESS"`
		CallerServerAddress string `env:"CALLER_SERVER_ADDRESS"`
		DatabaseType        string `env:"DATABASE_TYPE"`
		ListenerTimeout     int    `env:"LISTENER_TIMEOUT"`
	}
	Runtime struct {
		Type                string `env:"RUNTIME_TYPE"`
		AutoRemove          bool   `env:"RUNTIME_AUTOREMOVE"`
		Containerized       bool   `env:"RUNTIME_CONTAINERIZED"`
		FakeModelsPath      string `env:"FAKE_MODELS_PATH"`
		FakeTimeoutDuration int    `env:"FAKE_TIMEOUT_DURATION"`
		OnnxRuntimePath     string `env:"ONNX_RUNTIME_PATH"`
	}
	Log struct {
		Level    string `env:"LOG_LEVEL"`
		Format   string `env:"LOG_FORMAT"`
		FilePath string `env:"LOG_FILE"`
	}
	Stats struct {
		UpdateBufferSize int64 `env:"UPDATE_BUFFER_SIZE"`
	}
}

func parseArgs() (wc WorkerConfig) {
	flag.StringVar(&(wc.General.Address), "address", "", "Worker address. (Env: WORKER_ADDRESS)")
	flag.StringVar(&(wc.General.CallerServerAddress), "caller-server-address", "", "Caller server address. (Env: CALLER_SERVER_ADDRESS)")
	flag.StringVar(&(wc.General.DatabaseType), "database-type", "", "Type of the database. (Env: DATABASE_TYPE)")
	flag.StringVar(&(wc.Runtime.Type), "runtime", "docker", "Container runtime type. (Env: RUNTIME_TYPE)")
	flag.IntVar(&(wc.General.ListenerTimeout), "timeout", 20, "Timeout in seconds before leafnode listeners are removed from status stream updates. (Env: LISTENER_TIMEOUT)")
	flag.BoolVar(&(wc.Runtime.AutoRemove), "auto-remove", false, "Auto remove containers. (Env: RUNTIME_AUTOREMOVE)")
	flag.StringVar(&(wc.Log.Level), "log-level", "info", "Log level (debug, info, warn, error) (Env: LOG_LEVEL)")
	flag.StringVar(&(wc.Log.Format), "log-format", "text", "Log format (json or text) (Env: LOG_FORMAT)")
	flag.StringVar(&(wc.Log.FilePath), "log-file", "", "Log file path (defaults to stdout) (Env: LOG_FILE)")
	flag.BoolVar(&(wc.Runtime.Containerized), "containerized", false, "Use socket to connect to Docker. (Env: RUNTIME_CONTAINERIZED)")
	flag.StringVar(&(wc.Runtime.FakeModelsPath), "fake-models-path", "models.json", "Path to fake runtime models file. (Env: FAKE_MODELS_PATH)")
	flag.IntVar(&(wc.Runtime.FakeTimeoutDuration), "fake-timeout-duration", 30, "Fake container timeout duration in seconds. (Env: FAKE_TIMEOUT_DURATION)")
	flag.Int64Var(&(wc.Stats.UpdateBufferSize), "update-buffer-size", 10000, "Update buffer size. (Env: UPDATE_BUFFER_SIZE)")
	// The default value for this path is the installation path of the onnxruntime package in the debian worker container when installed with pip3
	flag.StringVar(&(wc.Runtime.OnnxRuntimePath), "onnxruntime-path", "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.22.0", "Path to the ONNX runtime. (Env: ONNX_RUNTIME_PATH)")
	flag.Parse()
	return
}

func setupLogger(config WorkerConfig) *slog.Logger {
	// Set up log level
	var level slog.Level
	switch config.Log.Level {
	case "debug":
		level = slog.LevelDebug
	case "info":
		level = slog.LevelInfo
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	// Set up log output
	var output *os.File
	var err error
	if config.Log.FilePath != "" {
		output, err = os.OpenFile(config.Log.FilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			slog.Error("Failed to open log file, falling back to stdout", "error", err)
			output = os.Stdout
		}
	} else {
		output = os.Stdout
	}

	// Set up handler options
	opts := &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
	}

	// Create handler based on format
	var handler slog.Handler
	if config.Log.Format == "json" {
		handler = slog.NewJSONHandler(output, opts)
	} else {
		handler = slog.NewTextHandler(output, opts)
	}

	// Create and set logger
	logger := slog.New(handler)
	slog.SetDefault(logger)

	return logger
}

func main() {
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()
	wc := parseArgs()
	logger := setupLogger(wc)

	logger.Info("Current configuration", "config", wc)

	statsManager := stats.NewStatsManager(logger, time.Duration(wc.General.ListenerTimeout)*time.Second, 1.0, wc.Stats.UpdateBufferSize)

	var dbAddress string
	var dbClient kv.FunctionMetadataStore

	if wc.Runtime.Containerized {
		dbAddress = "http://database:8999/" //needs to have this format for http to work
	} else {
		dbAddress = "http://localhost:8999"
	}

	switch wc.General.DatabaseType {
	case "http":
		dbClient = kv.NewHttpClient(dbAddress, logger)
	}

	var runtime cr.ContainerRuntime
	var callRouter network.CallRouterInterface

	switch wc.Runtime.Type {

	// normal, default runtime
	case "docker":
		runtime = dockerRuntime.NewDockerRuntime(wc.Runtime.Containerized, wc.Runtime.AutoRemove, wc.General.Address, logger)
		callRouter = network.NewCallRouter(logger)

	// Below are fake runtimes. To create new ones, add a new case here.
	// Fake runtimes need to provide the FunctionModels map to the FakeContainerRuntime constructor.
	case "fake-linear":
		// Load models from JSON file
		linearModels, err := fakeRuntime.LoadLinearModels(wc.Runtime.FakeModelsPath)

		if err != nil {
			logger.Error("Failed to load models", "error", err)
			os.Exit(1)
		}

		models := make(map[string]fakeRuntime.FunctionModel)
		for k, v := range linearModels {
			models[k] = &v
		}

		// Create fake runtime with loaded models
		fakeContainerRuntime := fakeRuntime.NewFakeContainerRuntime(
			logger,
			time.Duration(wc.Runtime.FakeTimeoutDuration)*time.Second,
			models)

		runtime = fakeContainerRuntime
		callRouter = fakeNetwork.NewFakeCallRouter(fakeContainerRuntime, logger)

		// fake runtime that returns the same values for all inputs. Useful for debugging.
	case "fake-instant":
		models := fakeRuntime.CreateInstantModels([]string{
			"hyperfaas-echo:latest",
			"hyperfaas-bfs-json:latest",
			"hyperfaas-thumbnailer-json:latest",
		})
		fakeContainerRuntime := fakeRuntime.NewFakeContainerRuntime(
			logger,
			time.Duration(wc.Runtime.FakeTimeoutDuration)*time.Second,
			models)
		runtime = fakeContainerRuntime
		callRouter = fakeNetwork.NewFakeCallRouter(fakeContainerRuntime, logger)

		// Uses the onnx runtime. Please make sure to set --onnxruntime-path to the correct path.
		// The model file names are hard coded for simplicity and assumed to be the ones below.
	case "fake-onnx":
		models, err := fakeRuntime.LoadOnnxModels(wc.Runtime.FakeModelsPath, map[string]string{
			"hyperfaas-echo:latest":             "echo.onnx",
			"hyperfaas-bfs-json:latest":         "bfs-json.onnx",
			"hyperfaas-thumbnailer-json:latest": "thumbnailer-json.onnx",
		}, wc.Runtime.OnnxRuntimePath)
		if err != nil {
			logger.Error("Failed to load models", "error", err)
			os.Exit(1)
		}
		fakeContainerRuntime := fakeRuntime.NewFakeContainerRuntime(logger, time.Duration(wc.Runtime.FakeTimeoutDuration)*time.Second, models)
		runtime = fakeContainerRuntime
		callRouter = fakeNetwork.NewFakeCallRouter(fakeContainerRuntime, logger)

	default:
		logger.Error("No runtime specified")
		os.Exit(1)
	}

	c := controller.NewController(runtime, statsManager, callRouter, logger, wc.General.Address, dbClient)

	c.StartServer()
}
