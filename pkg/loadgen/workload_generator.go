package loadgen

import (
	"math/rand/v2"
	"strconv"
	"time"
)

type WorkloadGenerator struct {
	seed        int64
	random      *rand.Rand
	maxDuration time.Duration
	patterns    map[string]*PhasePattern
	leafAddress string
	timeout     int32
}

type PhasePattern struct {
	ImageTag           string          `yaml:"image_tag"`
	PhaseCount         IntRange        `yaml:"phase_count"`
	ConstantLikelihood float64         `yaml:"constant_likelihood"` // 0.0-1.0
	RampingLikelihood  float64         `yaml:"ramping_likelihood"`  // 0.0-1.0
	Parameters         PhaseParameters `yaml:"parameters"`
}

type PhaseParameters struct {
	StartRPS IntRange `yaml:"start_rps"`
	EndRPS   IntRange `yaml:"end_rps"`
	Step     IntRange `yaml:"step"`
}

type IntRange struct {
	Min int `yaml:"min"`
	Max int `yaml:"max"`
}

func NewWorkloadGenerator(seed int64, maxDuration time.Duration, leafAddress string, timeout int32, patterns map[string]*PhasePattern) *WorkloadGenerator {
	return &WorkloadGenerator{
		seed:        seed,
		random:      rand.New(rand.NewPCG(uint64(seed), uint64(seed))),
		maxDuration: maxDuration,
		patterns:    patterns,
		leafAddress: leafAddress,
		timeout:     timeout,
	}
}

// Generates a workload for the given patterns. The phases for each function image tag will overlap.
func (g *WorkloadGenerator) GenerateWorkload() *Workload {
	workload := &Workload{
		Phases:      make([]TestPhase, 0),
		MaxDuration: g.maxDuration,
		LeafAddress: g.leafAddress,
		Timeout:     g.timeout,
	}

	times := make(map[string]time.Duration)
	for _, pattern := range g.patterns {
		times[pattern.ImageTag] = time.Duration(0)
	}

	for _, pattern := range g.patterns {
		phaseCount := g.getRandInt(pattern.PhaseCount.Min, pattern.PhaseCount.Max)
		phaseDuration := g.maxDuration / time.Duration(phaseCount)

		for i := 0; i < phaseCount; i++ {
			phaseStartTime := times[pattern.ImageTag]
			times[pattern.ImageTag] += phaseDuration
			c := g.random.Float64()
			var phaseType string
			if c < pattern.ConstantLikelihood {
				phaseType = "constant"
			} else {
				phaseType = "variable"
			}

			phase := TestPhase{
				Name:      pattern.ImageTag + "_" + strconv.Itoa(i),
				ImageTag:  pattern.ImageTag,
				Type:      phaseType,
				StartTime: phaseStartTime,
				Duration:  phaseDuration,
				StartRPS:  g.getRandInt(pattern.Parameters.StartRPS.Min, pattern.Parameters.StartRPS.Max),
				EndRPS:    g.getRandInt(pattern.Parameters.EndRPS.Min, pattern.Parameters.EndRPS.Max),
				Step:      g.getRandInt(pattern.Parameters.Step.Min, pattern.Parameters.Step.Max),
			}
			workload.Phases = append(workload.Phases, phase)
		}
	}

	return workload
}

func (g *WorkloadGenerator) getRandInt(min, max int) int {
	return g.random.IntN(max-min+1) + min
}
