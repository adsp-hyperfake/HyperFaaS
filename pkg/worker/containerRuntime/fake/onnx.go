package fake

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// Based on https://github.com/yalue/onnxruntime_go_examples/blob/master/sum_and_difference/sum_and_difference.go

// FunctionOnnxModel represents an ONNX neural network model for function prediction
type FunctionOnnxModel struct {
	ImageTag        string
	ModelPath       string
	session         *ort.AdvancedSession
	inputTensor     *ort.Tensor[float32]
	outputTensor    *ort.Tensor[float32]
	inputData       []float32
	mu              sync.Mutex
	initialized     bool
	onnxruntimePath string
}

// initializeOnnxRuntime initializes the ONNX runtime environment once
var onnxInitOnce sync.Once
var onnxInitialized bool

func (f *FunctionOnnxModel) ensureInitialized() error {
	if f.initialized {
		return nil
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if f.initialized {
		return nil
	}

	var err error
	// Initialize ONNX runtime environment once globally
	onnxInitOnce.Do(func() {

		if f.onnxruntimePath != "" {
			ort.SetSharedLibraryPath(f.onnxruntimePath)
			if err = ort.InitializeEnvironment(); err == nil {
				onnxInitialized = true
			}
		}
	})

	if !onnxInitialized {
		return fmt.Errorf("failed to initialize ONNX runtime environment, %w", err)
	}

	// Create input tensor for 5 features (body_size, instances, active_calls, worker_cpu, worker_ram)
	f.inputData = make([]float32, 4)
	f.inputTensor, err = ort.NewTensor(ort.NewShape(1, 4), f.inputData)
	if err != nil {
		return fmt.Errorf("error creating input tensor: %w", err)
	}

	// Create output tensor for 3 predictions (runtime, cpu, ram)
	f.outputTensor, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3))
	if err != nil {
		f.inputTensor.Destroy()
		return fmt.Errorf("error creating output tensor: %w", err)
	}

	// Create ONNX session
	f.session, err = ort.NewAdvancedSession(f.ModelPath,
		[]string{"input"},  // input name
		[]string{"output"}, // output name
		[]ort.ArbitraryTensor{f.inputTensor},
		[]ort.ArbitraryTensor{f.outputTensor},
		nil)
	if err != nil {
		f.inputTensor.Destroy()
		f.outputTensor.Destroy()
		return fmt.Errorf("error creating ONNX session: %w", err)
	}

	f.initialized = true
	return nil
}

// PredictFunction uses the ONNX model to predict function behavior
func (f *FunctionOnnxModel) PredictFunction(inputs FunctionInputs) (FunctionPrediction, error) {
	if err := f.ensureInitialized(); err != nil {
		return FunctionPrediction{}, err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	// Prepare input data: [body_size, instances, active_calls, worker_cpu, worker_ram]
	f.inputData[0] = float32(inputs.RequestBodySize)
	// f.inputData[1] = float32(inputs.FunctionInstances)
	f.inputData[1] = float32(inputs.ActiveFunctionCalls)
	f.inputData[2] = float32(inputs.WorkerCPUUsage)
	f.inputData[3] = float32(inputs.WorkerRAMUsage)

	// Run the model
	err := f.session.Run()
	if err != nil {
		return FunctionPrediction{}, fmt.Errorf("error running ONNX session: %w", err)
	}

	// Get output data
	outputData := f.outputTensor.GetData()
	if len(outputData) < 3 {
		return FunctionPrediction{}, fmt.Errorf("insufficient output data from ONNX model")
	}

	// Extract predictions (assuming output order: runtime_ms, cpu_usage, ram_usage)
	runtime := float64(outputData[0])
	cpuUsage := float64(outputData[1])
	ramUsage := float64(outputData[2])

	// Ensure positive values
	if runtime < 0 {
		runtime = 1.0 // 1ms minimum
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

// Cleanup destroys the ONNX session and tensors
func (f *FunctionOnnxModel) Cleanup() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.initialized {
		if f.session != nil {
			f.session.Destroy()
		}
		if f.inputTensor != nil {
			f.inputTensor.Destroy()
		}
		if f.outputTensor != nil {
			f.outputTensor.Destroy()
		}
		f.initialized = false
	}
}

// LoadOnnxModels loads ONNX models from a mapping of ImageTag to ModelFileName and returns a mapping of ImageTag to FunctionModel
func LoadOnnxModels(dir string, models map[string]string, onnxruntimePath string) (map[string]FunctionModel, error) {
	oModels := make(map[string]FunctionModel)
	for imageTag, modelFileName := range models {
		// Validate model file exists
		if _, err := os.Stat(filepath.Join(dir, modelFileName)); os.IsNotExist(err) {
			return nil, fmt.Errorf("ONNX model file does not exist: %s", modelFileName)
		}

		model := &FunctionOnnxModel{
			ImageTag:        imageTag,
			ModelPath:       filepath.Join(dir, modelFileName),
			onnxruntimePath: onnxruntimePath,
		}

		oModels[imageTag] = model
	}

	return oModels, nil
}
