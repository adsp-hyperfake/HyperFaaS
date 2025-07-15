package fake

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"

	cr "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime"
	"github.com/3s-rg-codes/HyperFaaS/proto/common"
	"github.com/3s-rg-codes/HyperFaaS/proto/controller"
	"github.com/google/uuid"
)

// FunctionModel represents a linear model for a specific function image tag
type FunctionModel struct {
	ImageTag      string     `json:"image_tag"`
	RuntimeCoeffs [6]float64 `json:"runtime_coeffs"` // [body_size, instances, active_calls, worker_cpu, worker_ram, intercept]
	CPUCoeffs     [6]float64 `json:"cpu_coeffs"`     // same order
	RAMCoeffs     [6]float64 `json:"ram_coeffs"`     // same order
	SampleCount   int        `json:"sample_count"`
}

// FakeContainer represents a simulated container instance
type FakeContainer struct {
	InstanceID      string
	InstanceIP      string
	InstanceName    string
	ImageTag        string
	FunctionID      string
	StartTime       time.Time
	LastRequestTime time.Time
	IsRunning       bool
	CancelFunc      context.CancelFunc
	TimeoutDuration time.Duration
}

// FakeContainerRuntime simulates container operations using mathematical models
type FakeContainerRuntime struct {
	cr.ContainerRuntime
	models          map[string]*FunctionModel
	containers      map[string]*FakeContainer
	mu              sync.RWMutex
	logger          *slog.Logger
	timeoutDuration time.Duration
	rng             *rand.Rand
}

// NewFakeContainerRuntime creates a new fake container runtime by loading models from file
func NewFakeContainerRuntime(modelsPath string, timeoutDuration time.Duration, logger *slog.Logger) (*FakeContainerRuntime, error) {
	models, err := loadModels(modelsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load models: %v", err)
	}

	return NewFakeContainerRuntimeWithModels(models, timeoutDuration, logger)
}

// NewFakeContainerRuntimeWithModels creates a new fake container runtime with pre-loaded models
func NewFakeContainerRuntimeWithModels(models map[string]*FunctionModel, timeoutDuration time.Duration, logger *slog.Logger) (*FakeContainerRuntime, error) {
	runtime := &FakeContainerRuntime{
		models:          models,
		containers:      make(map[string]*FakeContainer),
		logger:          logger,
		timeoutDuration: timeoutDuration,
		rng:             rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	logger.Info("Fake container runtime initialized", "models_count", len(models), "timeout_duration", timeoutDuration)

	return runtime, nil
}

func loadModels(modelsPath string) (map[string]*FunctionModel, error) {
	file, err := os.Open(modelsPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var models map[string]*FunctionModel
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&models)
	if err != nil {
		return nil, err
	}

	return models, nil
}

// Start simulates starting a container
func (f *FakeContainerRuntime) Start(ctx context.Context, functionID string, imageTag string, config *common.Config) (cr.Container, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Generate instance details
	longID := uuid.New().String()
	shortID := longID[:12]
	instanceName := fmt.Sprintf("fake-%s-%s", strings.ReplaceAll(imageTag, ":", "-"), shortID[:8])
	instanceIP := fmt.Sprintf("fake-%s:50052", shortID)

	// Create fake container
	containerCtx, cancel := context.WithCancel(context.Background())
	container := &FakeContainer{
		InstanceID:      shortID,
		InstanceIP:      instanceIP,
		InstanceName:    instanceName,
		ImageTag:        imageTag,
		FunctionID:      functionID,
		StartTime:       time.Now(),
		LastRequestTime: time.Now(),
		IsRunning:       true,
		CancelFunc:      cancel,
		TimeoutDuration: f.timeoutDuration,
	}

	f.containers[shortID] = container

	// Start timeout monitoring goroutine
	go f.monitorTimeout(containerCtx, container)

	f.logger.Debug("Started fake container",
		"functionID", functionID,
		"instanceID", shortID,
		"imageTag", imageTag,
		"instanceIP", instanceIP,
		"timeout", f.timeoutDuration,
	)

	return cr.Container{
		InstanceID:   shortID,
		InstanceIP:   instanceIP,
		InstanceName: instanceName,
	}, nil
}

// monitorTimeout simulates container timeout behavior
func (f *FakeContainerRuntime) monitorTimeout(ctx context.Context, container *FakeContainer) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			f.logger.Debug("Container monitoring stopped", "instanceID", container.InstanceID)
			return
		case <-ticker.C:
			f.mu.RLock()
			if !container.IsRunning {
				f.mu.RUnlock()
				return
			}

			timeSinceLastRequest := time.Since(container.LastRequestTime)
			shouldTimeout := timeSinceLastRequest > container.TimeoutDuration
			f.mu.RUnlock()

			if shouldTimeout {
				f.logger.Debug("Container timed out",
					"instanceID", container.InstanceID,
					"timeSinceLastRequest", timeSinceLastRequest,
				)
				f.handleContainerTimeout(container)
				return
			}
		}
	}
}

func (f *FakeContainerRuntime) handleContainerTimeout(container *FakeContainer) {
	f.mu.Lock()
	container.IsRunning = false
	container.CancelFunc()
	f.mu.Unlock()
}

// Call is not used in the current architecture, but required by interface
func (f *FakeContainerRuntime) Call(ctx context.Context, req *common.CallRequest) (*common.CallResponse, error) {
	return nil, fmt.Errorf("Call method not implemented in fake runtime")
}

// Stop simulates stopping a container
func (f *FakeContainerRuntime) Stop(ctx context.Context, req *common.InstanceID) (*common.InstanceID, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	container, exists := f.containers[req.Id]
	if !exists {
		return nil, fmt.Errorf("container not found: %s", req.Id)
	}

	if container.IsRunning {
		container.IsRunning = false
		container.CancelFunc()
		f.logger.Debug("Stopped fake container", "instanceID", req.Id)
	}

	return req, nil
}

// Status is not implemented but required by interface
func (f *FakeContainerRuntime) Status(req *controller.StatusRequest, stream controller.Controller_StatusServer) error {
	return fmt.Errorf("Status method not implemented in fake runtime")
}

// MonitorContainer simulates container lifecycle monitoring
func (f *FakeContainerRuntime) MonitorContainer(ctx context.Context, instanceId *common.InstanceID, functionId string) error {
	f.mu.RLock()
	container, exists := f.containers[instanceId.Id]
	f.mu.RUnlock()

	if !exists {
		return fmt.Errorf("container not found: %s", instanceId.Id)
	}

	// Wait for container to stop (either via timeout or manual stop)
	for {
		f.mu.RLock()
		isRunning := container.IsRunning
		f.mu.RUnlock()

		if !isRunning {
			f.logger.Debug("Container monitoring completed", "instanceID", instanceId.Id)
			return nil // Return nil for timeout (graceful shutdown)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
			// Continue monitoring
		}
	}
}

// RemoveImage simulates image removal (no-op for fake runtime)
func (f *FakeContainerRuntime) RemoveImage(ctx context.Context, imageID string) error {
	f.logger.Debug("RemoveImage called (no-op)", "imageID", imageID)
	return nil
}

// ContainerExists checks if a fake container exists
func (f *FakeContainerRuntime) ContainerExists(ctx context.Context, instanceID string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	_, exists := f.containers[instanceID]
	return exists
}

// ContainerStats returns fake stats (returns empty reader)
func (f *FakeContainerRuntime) ContainerStats(ctx context.Context, containerID string) io.ReadCloser {
	return io.NopCloser(strings.NewReader("{}"))
}

// PredictFunction uses the mathematical model to predict function behavior
func (f *FakeContainerRuntime) PredictFunction(imageTag string, inputs FunctionInputs) (FunctionPrediction, error) {
	f.mu.RLock()
	model, exists := f.models[imageTag]
	f.mu.RUnlock()

	if !exists {
		// Return default values if no model exists
		return FunctionPrediction{
			Runtime:  time.Duration(100 * time.Millisecond), // Default runtime
			CPUUsage: 0.1,                                   // Default CPU usage
			RAMUsage: 50 * 1024 * 1024,                      // Default 50MB RAM
		}, nil
	}

	// Prepare feature vector: [body_size, instances, active_calls, worker_cpu, worker_ram, 1.0]
	features := [6]float64{
		float64(inputs.RequestBodySize),
		float64(inputs.FunctionInstances),
		float64(inputs.ActiveFunctionCalls),
		inputs.WorkerCPUUsage,
		float64(inputs.WorkerRAMUsage),
		1.0, // intercept
	}

	// Predict using linear models
	runtime := f.predict(model.RuntimeCoeffs, features)
	cpuUsage := f.predict(model.CPUCoeffs, features)
	ramUsage := f.predict(model.RAMCoeffs, features)

	// Add some gaussian noise for realism (±10%)
	runtime = f.addNoise(runtime, 0.1)
	cpuUsage = f.addNoise(cpuUsage, 0.1)
	ramUsage = f.addNoise(ramUsage, 0.1)

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
		Runtime:  time.Duration(runtime) * time.Nanosecond,
		CPUUsage: cpuUsage,
		RAMUsage: int64(ramUsage),
	}, nil
}

func (f *FakeContainerRuntime) predict(coeffs [6]float64, features [6]float64) float64 {
	result := 0.0
	for i := 0; i < 6; i++ {
		result += coeffs[i] * features[i]
	}
	return result
}

func (f *FakeContainerRuntime) addNoise(value float64, noiseLevel float64) float64 {
	noise := f.rng.NormFloat64() * noiseLevel * value
	return value + noise
}

// UpdateLastRequestTime updates the last request time for a container (prevents timeout)
func (f *FakeContainerRuntime) UpdateLastRequestTime(instanceID string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if container, exists := f.containers[instanceID]; exists {
		container.LastRequestTime = time.Now()
	}
}

// GetRunningContainers returns the current running containers count
func (f *FakeContainerRuntime) GetRunningContainers() map[string]*FakeContainer {
	f.mu.RLock()
	defer f.mu.RUnlock()

	result := make(map[string]*FakeContainer)
	for id, container := range f.containers {
		if container.IsRunning {
			result[id] = container
		}
	}
	return result
}

// FunctionInputs represents the input parameters for function prediction
type FunctionInputs struct {
	RequestBodySize     int64
	FunctionInstances   int64
	ActiveFunctionCalls int64
	WorkerCPUUsage      float64
	WorkerRAMUsage      int64
}

// FunctionPrediction represents the predicted function behavior
type FunctionPrediction struct {
	Runtime  time.Duration
	CPUUsage float64
	RAMUsage int64
}
