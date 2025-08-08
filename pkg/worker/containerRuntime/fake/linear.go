package fake

import (
	"encoding/json"
	"os"
	"time"
)

// FunctionLinearModel represents a linear model for a specific function image tag
type FunctionLinearModel struct {
	ImageTag      string     `json:"image_tag"`
	RuntimeCoeffs [6]float64 `json:"runtime_coeffs"` // [body_size, instances, active_calls, worker_cpu, worker_ram, intercept]
	CPUCoeffs     [6]float64 `json:"cpu_coeffs"`     // same order
	RAMCoeffs     [6]float64 `json:"ram_coeffs"`     // same order
	SampleCount   int        `json:"sample_count"`
}

// LoadLinearModels loads a JSON file containing linear models, and returns a mapping of ImageTag to the model. It cannot be a mapping of functionID to the model, because the functionID is not known at startup.
func LoadLinearModels(modelsPath string) (map[string]FunctionLinearModel, error) {
	file, err := os.Open(modelsPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var models map[string]FunctionLinearModel
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&models)
	if err != nil {
		return nil, err
	}

	return models, nil
}

// PredictFunction uses the mathematical model to predict function behavior
func (f *FunctionLinearModel) PredictFunction(inputs FunctionInputs) (FunctionPrediction, error) {
	// Prepare feature vector: [body_size, instances, active_calls, worker_cpu, worker_ram, 1.0]
	features := [6]float64{
		float64(inputs.RequestBodySize),
		float64(inputs.FunctionInstances),
		float64(inputs.ActiveFunctionCalls),
		float64(inputs.WorkerCPUUsage),
		float64(inputs.WorkerRAMUsage),
		1.0, // intercept
	}

	// Predict using linear models
	runtime := f.predict(f.RuntimeCoeffs, features)
	cpuUsage := f.predict(f.CPUCoeffs, features)
	ramUsage := f.predict(f.RAMCoeffs, features)

	// Ensure positive values
	if runtime < 0 {
		runtime = 1000000 // 1ms minimum
	}
	if cpuUsage < 0 {
		cpuUsage = 0
	}
	if ramUsage < 0 {
		ramUsage = 1024 * 1024 // 1MB minimum
	}

	return FunctionPrediction{
		Runtime:  time.Duration(runtime) * time.Millisecond,
		CPUUsage: int64(cpuUsage),
		RAMUsage: int64(ramUsage),
	}, nil
}

func (f *FunctionLinearModel) predict(coeffs [6]float64, features [6]float64) float64 {
	result := 0.0
	for i := 0; i < 6; i++ {
		result += coeffs[i] * features[i]
	}
	return result
}
