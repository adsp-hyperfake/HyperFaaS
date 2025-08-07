package fake

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"time"

	"sync/atomic"

	cr "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime"
	"github.com/3s-rg-codes/HyperFaaS/proto/common"
	"github.com/3s-rg-codes/HyperFaaS/proto/controller"
	"github.com/google/uuid"
)

// FunctionInputs represents the input parameters for function prediction
type FunctionInputs struct {
	RequestBodySize     int64
	FunctionInstances   int32
	ActiveFunctionCalls int32
	WorkerCPUUsage      int64
	WorkerRAMUsage      int64
}

// FunctionPrediction represents the predicted function behavior
type FunctionPrediction struct {
	Runtime  time.Duration
	CPUUsage int64
	RAMUsage int64
}

type FunctionModel interface {
	PredictFunction(inputs FunctionInputs) (FunctionPrediction, error)
}

// FakeContainer represents a simulated container instance
type FakeContainer struct {
	InstanceID      string
	InstanceIP      string
	InstanceName    string
	ImageTag        string
	FunctionID      string
	RequestChan     chan struct{}
	StartTime       time.Time
	LastRequestTime time.Time
	CancelFunc      context.CancelFunc
	TimeoutDuration time.Duration
	Context         context.Context
}

// Run simulates the container running and waiting for requests. After the timeout duration without receiving a request, the container is stopped and its context is cancelled.
func (c *FakeContainer) Run() {
	timeout := time.Now().Add(c.TimeoutDuration)
	for {
		select {
		// not super proud of this implementation given that this receives potentially a lot of requests
		// but it's a simple way to simulate the container running and waiting for requests.
		case <-c.RequestChan:
			timeout = time.Now().Add(c.TimeoutDuration)
		case <-time.After(time.Until(timeout)):
			c.CancelFunc()
			return
		}
	}
}

// FakeContainers is a wrapper for the containers map
type FakeContainers struct {
	Containers map[string]*FakeContainer
	Mu         sync.RWMutex
}

// FakeModels is a wrapper for the models map
type FakeModels struct {
	Models map[string]FunctionModel
	Mu     sync.RWMutex
}

// FakeLinearContainerRuntime simulates container operations using mathematical models
type FakeContainerRuntime struct {
	Containers               FakeContainers
	Models                   FakeModels
	CurrentCPUUsage          atomic.Int64
	CurrentRAMUsage          atomic.Int64
	CurrentActiveCalls       atomic.Int32
	CurrentRunningContainers atomic.Int32
	Logger                   *slog.Logger
	TimeoutDuration          time.Duration
}

// Start simulates starting a container
func (f *FakeContainerRuntime) Start(ctx context.Context, functionID string, imageTag string, config *common.Config) (cr.Container, error) {

	longID := uuid.New().String()
	shortID := longID[:12]
	instanceName := fmt.Sprintf("fake-%s-%s", strings.ReplaceAll(imageTag, ":", "-"), shortID[:8])
	instanceIP := fmt.Sprintf("fake-%s:50052", shortID)

	cCtx, cancel := context.WithTimeout(context.Background(), f.TimeoutDuration)
	container := &FakeContainer{
		InstanceID:      shortID,
		InstanceIP:      instanceIP,
		InstanceName:    instanceName,
		ImageTag:        imageTag,
		FunctionID:      functionID,
		StartTime:       time.Now(),
		LastRequestTime: time.Now(),
		CancelFunc:      cancel,
		TimeoutDuration: f.TimeoutDuration,
		Context:         cCtx,
		RequestChan:     make(chan struct{}),
	}
	go container.Run()

	f.Containers.Mu.Lock()
	f.Containers.Containers[shortID] = container
	f.Containers.Mu.Unlock()

	f.Logger.Debug("Started fake container",
		"functionID", functionID,
		"instanceID", shortID,
		"imageTag", imageTag,
		"instanceIP", instanceIP,
		"timeout", f.TimeoutDuration,
	)

	f.CurrentRunningContainers.Add(1)

	// Find a better way to simulate this ??
	time.Sleep(20 * time.Millisecond)

	return cr.Container{
		InstanceID:   shortID,
		InstanceIP:   instanceIP,
		InstanceName: instanceName,
	}, nil
}

// Call uses a model to predict function behavior and returns a response
func (f *FakeContainerRuntime) Call(ctx context.Context, req *common.CallRequest) (*common.CallResponse, error) {
	f.Containers.Mu.RLock()
	container, exists := f.Containers.Containers[req.InstanceId.Id]
	f.Containers.Mu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("container not found: %s", req.InstanceId.Id)
	}
	// reset container timeout
	if container.RequestChan != nil {
		container.RequestChan <- struct{}{}
	} else {
		return nil, fmt.Errorf("container not found: %s", req.InstanceId.Id)
	}

	f.Models.Mu.RLock()
	model, exists := f.Models.Models[req.FunctionId.Id]
	f.Models.Mu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("model not found for function ID: %s", req.FunctionId)
	}

	inputs := FunctionInputs{
		RequestBodySize:     int64(len(req.Data)),
		FunctionInstances:   f.CurrentRunningContainers.Load(),
		ActiveFunctionCalls: f.CurrentActiveCalls.Load(),
		WorkerCPUUsage:      f.CurrentCPUUsage.Load(),
		WorkerRAMUsage:      f.CurrentRAMUsage.Load(),
	}

	f.CurrentActiveCalls.Add(1)
	prediction, err := model.PredictFunction(inputs)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %v", err)
	}
	f.CurrentCPUUsage.Add(prediction.CPUUsage)
	f.CurrentRAMUsage.Add(prediction.RAMUsage)
	time.Sleep(prediction.Runtime)

	// release after call has been processed
	f.CurrentActiveCalls.Add(-1)
	f.CurrentCPUUsage.Add(-prediction.CPUUsage)
	f.CurrentRAMUsage.Add(-prediction.RAMUsage)

	return &common.CallResponse{
		Data: []byte(""),
	}, nil
}

// Stop simulates stopping a container
func (f *FakeContainerRuntime) Stop(ctx context.Context, req *common.InstanceID) (*common.InstanceID, error) {
	f.Containers.Mu.Lock()
	container, exists := f.Containers.Containers[req.Id]
	f.Containers.Mu.Unlock()

	if !exists {
		return nil, fmt.Errorf("container not found: %s", req.Id)
	}

	container.CancelFunc()
	f.Logger.Debug("Stopped fake container", "instanceID", req.Id)

	return req, nil
}

// Status is not implemented but required by interface
func (f *FakeContainerRuntime) Status(req *controller.StatusRequest, stream controller.Controller_StatusServer) error {
	return fmt.Errorf("Status method not implemented in fake runtime")
}

// MonitorContainer simulates container lifecycle monitoring
func (f *FakeContainerRuntime) MonitorContainer(ctx context.Context, instanceId *common.InstanceID, functionId string) error {
	f.Containers.Mu.RLock()
	container, exists := f.Containers.Containers[instanceId.Id]
	f.Containers.Mu.RUnlock()

	if !exists {
		return fmt.Errorf("container not found: %s", instanceId.Id)
	}

	// Wait for container to stop (either via timeout or manual stop)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-container.Context.Done():
			f.Logger.Debug("Container monitoring completed", "instanceID", instanceId.Id)
			// delete it from the map
			f.Containers.Mu.Lock()
			delete(f.Containers.Containers, instanceId.Id)
			f.Containers.Mu.Unlock()

			return nil
		}
	}
}

func (f *FakeContainerRuntime) RemoveImage(ctx context.Context, imageID string) error {
	return nil
}

// ContainerExists checks if a fake container exists
func (f *FakeContainerRuntime) ContainerExists(ctx context.Context, instanceID string) bool {
	f.Containers.Mu.RLock()
	defer f.Containers.Mu.RUnlock()
	_, exists := f.Containers.Containers[instanceID]
	return exists
}

// ContainerStats returns fake stats (returns empty reader)
func (f *FakeContainerRuntime) ContainerStats(ctx context.Context, containerID string) io.ReadCloser {
	return io.NopCloser(strings.NewReader("{}"))
}
