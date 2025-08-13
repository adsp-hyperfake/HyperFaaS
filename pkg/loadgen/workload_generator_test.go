package loadgen

import (
	"testing"
	"time"
)

func TestWorkloadGenerator_GenerateWorkload(t *testing.T) {
	tests := []struct {
		name        string
		seed        int64
		maxDuration time.Duration
		patterns    map[string]*PhasePattern
		validate    func(t *testing.T, workload *Workload, patterns map[string]*PhasePattern, maxDuration time.Duration)
	}{
		{
			name:        "single_pattern_single_phase_constant_only",
			seed:        1,
			maxDuration: 10 * time.Second,
			patterns: map[string]*PhasePattern{
				"hyperfaas-echo:latest": {
					ImageTag: "hyperfaas-echo:latest",
					PhaseCount: IntRange{
						Min: 1,
						Max: 1,
					},
					ConstantLikelihood: 1.0,
					RampingLikelihood:  0.0,
					Parameters: PhaseParameters{
						StartRPS: IntRange{Min: 10, Max: 20},
						EndRPS:   IntRange{Min: 30, Max: 40},
						Step:     IntRange{Min: 1, Max: 5},
					},
				},
			},
			validate: validateSinglePatternSinglePhase,
		},
		{
			name:        "single_pattern_multiple_phases_mixed_types",
			seed:        42,
			maxDuration: 30 * time.Second,
			patterns: map[string]*PhasePattern{
				"test-function:v1": {
					ImageTag: "test-function:v1",
					PhaseCount: IntRange{
						Min: 3,
						Max: 5,
					},
					ConstantLikelihood: 0.6,
					RampingLikelihood:  0.4,
					Parameters: PhaseParameters{
						StartRPS: IntRange{Min: 5, Max: 15},
						EndRPS:   IntRange{Min: 20, Max: 50},
						Step:     IntRange{Min: 2, Max: 8},
					},
				},
			},
			validate: validateMultiplePhases,
		},
		{
			name:        "multiple_patterns_overlapping_phases",
			seed:        123,
			maxDuration: 60 * time.Second,
			patterns: map[string]*PhasePattern{
				"function-a:latest": {
					ImageTag: "function-a:latest",
					PhaseCount: IntRange{
						Min: 2,
						Max: 3,
					},
					ConstantLikelihood: 0.8,
					RampingLikelihood:  0.2,
					Parameters: PhaseParameters{
						StartRPS: IntRange{Min: 1, Max: 10},
						EndRPS:   IntRange{Min: 11, Max: 25},
						Step:     IntRange{Min: 1, Max: 3},
					},
				},
				"function-b:v2": {
					ImageTag: "function-b:v2",
					PhaseCount: IntRange{
						Min: 1,
						Max: 2,
					},
					ConstantLikelihood: 0.3,
					RampingLikelihood:  0.7,
					Parameters: PhaseParameters{
						StartRPS: IntRange{Min: 50, Max: 100},
						EndRPS:   IntRange{Min: 100, Max: 200},
						Step:     IntRange{Min: 5, Max: 15},
					},
				},
			},
			validate: validateMultiplePatterns,
		},
		{
			name:        "edge_case_ramping_only",
			seed:        999,
			maxDuration: 5 * time.Second,
			patterns: map[string]*PhasePattern{
				"ramping-only:test": {
					ImageTag: "ramping-only:test",
					PhaseCount: IntRange{
						Min: 2,
						Max: 2,
					},
					ConstantLikelihood: 0.0,
					RampingLikelihood:  1.0,
					Parameters: PhaseParameters{
						StartRPS: IntRange{Min: 1, Max: 1},
						EndRPS:   IntRange{Min: 100, Max: 100},
						Step:     IntRange{Min: 10, Max: 10},
					},
				},
			},
			validate: validateRampingOnly,
		},
		{
			name:        "edge_case_single_value_ranges",
			seed:        777,
			maxDuration: 15 * time.Second,
			patterns: map[string]*PhasePattern{
				"single-values:test": {
					ImageTag: "single-values:test",
					PhaseCount: IntRange{
						Min: 3,
						Max: 3,
					},
					ConstantLikelihood: 0.5,
					RampingLikelihood:  0.5,
					Parameters: PhaseParameters{
						StartRPS: IntRange{Min: 25, Max: 25},
						EndRPS:   IntRange{Min: 75, Max: 75},
						Step:     IntRange{Min: 2, Max: 2},
					},
				},
			},
			validate: validateSingleValueRanges,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewWorkloadGenerator(tt.seed, tt.maxDuration, "localhost:50050", 10, tt.patterns)
			workload := generator.GenerateWorkload()

			if workload == nil {
				t.Fatal("GenerateWorkload() returned nil workload")
			}

			if workload.Phases == nil {
				t.Fatal("Generated workload has nil Phases")
			}

			tt.validate(t, workload, tt.patterns, tt.maxDuration)
		})
	}
}

func validateSinglePatternSinglePhase(t *testing.T, workload *Workload, patterns map[string]*PhasePattern, maxDuration time.Duration) {
	if len(workload.Phases) != 1 {
		t.Errorf("Expected exactly 1 phase, got %d", len(workload.Phases))
		return
	}

	phase := workload.Phases[0]
	pattern := patterns["hyperfaas-echo:latest"]

	if phase.ImageTag != pattern.ImageTag {
		t.Errorf("Expected ImageTag %s, got %s", pattern.ImageTag, phase.ImageTag)
	}

	if phase.Type != "constant" {
		t.Errorf("Expected phase type 'constant' with 100%% constant likelihood, got %s", phase.Type)
	}

	if phase.StartTime != 0 {
		t.Errorf("Expected first phase StartTime to be 0, got %v", phase.StartTime)
	}

	if phase.Duration != maxDuration {
		t.Errorf("Expected phase Duration to be %v, got %v", maxDuration, phase.Duration)
	}

	validateParameterRanges(t, phase, pattern.Parameters)
}

func validateMultiplePhases(t *testing.T, workload *Workload, patterns map[string]*PhasePattern, maxDuration time.Duration) {
	pattern := patterns["test-function:v1"]
	phaseCount := len(workload.Phases)

	if phaseCount < pattern.PhaseCount.Min || phaseCount > pattern.PhaseCount.Max {
		t.Errorf("Expected phase count between %d and %d, got %d",
			pattern.PhaseCount.Min, pattern.PhaseCount.Max, phaseCount)
		return
	}

	expectedPhaseDuration := maxDuration / time.Duration(phaseCount)

	for i, phase := range workload.Phases {
		if phase.ImageTag != pattern.ImageTag {
			t.Errorf("Phase %d: Expected ImageTag %s, got %s", i, pattern.ImageTag, phase.ImageTag)
		}

		if phase.Type != "constant" && phase.Type != "variable" {
			t.Errorf("Phase %d: Expected type 'constant' or 'variable', got %s", i, phase.Type)
		}

		expectedStartTime := time.Duration(i) * expectedPhaseDuration
		if phase.StartTime != expectedStartTime {
			t.Errorf("Phase %d: Expected StartTime %v, got %v", i, expectedStartTime, phase.StartTime)
		}

		if phase.Duration != expectedPhaseDuration {
			t.Errorf("Phase %d: Expected Duration %v, got %v", i, expectedPhaseDuration, phase.Duration)
		}

		validateParameterRanges(t, phase, pattern.Parameters)
	}
}

func validateMultiplePatterns(t *testing.T, workload *Workload, patterns map[string]*PhasePattern, maxDuration time.Duration) {
	phaseCounts := make(map[string]int)

	for _, phase := range workload.Phases {
		phaseCounts[phase.ImageTag]++
	}

	for imageTag, pattern := range patterns {
		count := phaseCounts[imageTag]
		if count < pattern.PhaseCount.Min || count > pattern.PhaseCount.Max {
			t.Errorf("Pattern %s: Expected phase count between %d and %d, got %d",
				imageTag, pattern.PhaseCount.Min, pattern.PhaseCount.Max, count)
		}
	}

	// Validate each phase
	for i, phase := range workload.Phases {
		pattern, exists := patterns[phase.ImageTag]
		if !exists {
			t.Errorf("Phase %d: Unknown ImageTag %s", i, phase.ImageTag)
			continue
		}

		if phase.Type != "constant" && phase.Type != "variable" {
			t.Errorf("Phase %d: Expected type 'constant' or 'variable', got %s", i, phase.Type)
		}

		validateParameterRanges(t, phase, pattern.Parameters)
	}

	// Validate timing for each pattern
	for imageTag, _ := range patterns {
		phases := getPhasesByImageTag(workload.Phases, imageTag)
		if len(phases) == 0 {
			t.Errorf("No phases found for pattern %s", imageTag)
			continue
		}

		expectedPhaseDuration := maxDuration / time.Duration(len(phases))
		for i, phase := range phases {
			expectedStartTime := time.Duration(i) * expectedPhaseDuration
			if phase.StartTime != expectedStartTime {
				t.Errorf("Pattern %s, Phase %d: Expected StartTime %v, got %v",
					imageTag, i, expectedStartTime, phase.StartTime)
			}
		}
	}
}

func validateRampingOnly(t *testing.T, workload *Workload, patterns map[string]*PhasePattern, maxDuration time.Duration) {
	if len(workload.Phases) != 2 {
		t.Errorf("Expected exactly 2 phases, got %d", len(workload.Phases))
		return
	}

	pattern := patterns["ramping-only:test"]

	for i, phase := range workload.Phases {
		if phase.Type != "variable" {
			t.Errorf("Phase %d: Expected type 'variable' with 100%% ramping likelihood, got %s", i, phase.Type)
		}

		validateParameterRanges(t, phase, pattern.Parameters)
	}
}

func validateSingleValueRanges(t *testing.T, workload *Workload, patterns map[string]*PhasePattern, maxDuration time.Duration) {
	if len(workload.Phases) != 3 {
		t.Errorf("Expected exactly 3 phases, got %d", len(workload.Phases))
		return
	}

	for i, phase := range workload.Phases {
		// With single value ranges, all values should be exactly the min/max value
		if phase.StartRPS != 25 {
			t.Errorf("Phase %d: Expected StartRPS 25, got %d", i, phase.StartRPS)
		}
		if phase.EndRPS != 75 {
			t.Errorf("Phase %d: Expected EndRPS 75, got %d", i, phase.EndRPS)
		}
		if phase.Step != 2 {
			t.Errorf("Phase %d: Expected Step 2, got %d", i, phase.Step)
		}
	}
}

func validateParameterRanges(t *testing.T, phase TestPhase, params PhaseParameters) {
	if phase.StartRPS < params.StartRPS.Min || phase.StartRPS > params.StartRPS.Max {
		t.Errorf("StartRPS %d is outside expected range [%d, %d]",
			phase.StartRPS, params.StartRPS.Min, params.StartRPS.Max)
	}

	if phase.EndRPS < params.EndRPS.Min || phase.EndRPS > params.EndRPS.Max {
		t.Errorf("EndRPS %d is outside expected range [%d, %d]",
			phase.EndRPS, params.EndRPS.Min, params.EndRPS.Max)
	}

	if phase.Step < params.Step.Min || phase.Step > params.Step.Max {
		t.Errorf("Step %d is outside expected range [%d, %d]",
			phase.Step, params.Step.Min, params.Step.Max)
	}
}

func getPhasesByImageTag(phases []TestPhase, imageTag string) []TestPhase {
	var result []TestPhase
	for _, phase := range phases {
		if phase.ImageTag == imageTag {
			result = append(result, phase)
		}
	}
	return result
}
