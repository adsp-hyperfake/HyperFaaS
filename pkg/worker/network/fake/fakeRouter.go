package fake

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	fakeRuntime "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime/fake"
	"github.com/3s-rg-codes/HyperFaaS/proto/common"
)

// FakeCallRouter simulates function calls using mathematical models
type FakeCallRouter struct {
	mu                    sync.RWMutex
	functionInstances     map[string][]string // functionID -> list of instanceIPs
	runtime               *fakeRuntime.FakeContainerRuntime
	logger                *slog.Logger
	activeCallCounts      map[string]int // functionID -> active call count
	globalActiveCallCount int
}

// NewFakeCallRouter creates a new fake call router
func NewFakeCallRouter(runtime *fakeRuntime.FakeContainerRuntime, logger *slog.Logger) *FakeCallRouter {
	return &FakeCallRouter{
		functionInstances:     make(map[string][]string),
		runtime:               runtime,
		logger:                logger,
		activeCallCounts:      make(map[string]int),
		globalActiveCallCount: 0,
	}
}

// CallFunction simulates calling a function using mathematical models
func (f *FakeCallRouter) CallFunction(functionID string, req *common.CallRequest) (*common.CallResponse, error) {
	f.mu.Lock()
	instances, exists := f.functionInstances[functionID]
	if !exists || len(instances) == 0 {
		f.mu.Unlock()
		return nil, fmt.Errorf("no instances available for function: %s", functionID)
	}

	// Increment active call counts
	f.activeCallCounts[functionID]++
	f.globalActiveCallCount++

	// Get current metrics for prediction
	runningContainers := f.runtime.GetRunningContainers()
	functionInstanceCount := len(instances)
	activeCalls := f.activeCallCounts[functionID]

	f.mu.Unlock()

	// Simulate selecting an instance (round-robin would happen here in real implementation)
	selectedInstance := instances[0]

	// Extract instance ID from IP format (fake-instanceID:50052)
	instanceID := extractInstanceIDFromIP(selectedInstance)

	// Get image tag for the selected instance
	imageTag := f.getImageTagForInstance(instanceID, runningContainers)

	// Prepare inputs for prediction
	inputs := fakeRuntime.FunctionInputs{
		RequestBodySize:     int64(len(req.Data)),
		FunctionInstances:   int64(functionInstanceCount),
		ActiveFunctionCalls: int64(activeCalls),
		WorkerCPUUsage:      f.getCurrentWorkerCPU(),
		WorkerRAMUsage:      f.getCurrentWorkerRAM(),
	}

	f.logger.Debug("Predicting function call",
		"functionID", functionID,
		"instanceID", instanceID,
		"imageTag", imageTag,
		"bodySize", inputs.RequestBodySize,
		"instances", inputs.FunctionInstances,
		"activeCalls", inputs.ActiveFunctionCalls,
	)

	// Use model to predict function behavior
	prediction, err := f.runtime.PredictFunction(imageTag, inputs)
	if err != nil {
		f.mu.Lock()
		f.activeCallCounts[functionID]--
		f.globalActiveCallCount--
		f.mu.Unlock()
		return nil, fmt.Errorf("prediction failed: %v", err)
	}

	// Simulate the actual function execution time
	time.Sleep(prediction.Runtime)

	// Update the container's last request time to prevent timeout
	f.runtime.UpdateLastRequestTime(instanceID)

	// Decrement active call counts
	f.mu.Lock()
	f.activeCallCounts[functionID]--
	f.globalActiveCallCount--
	f.mu.Unlock()

	// Generate a fake response based on the function type
	responseData := f.generateFakeResponse(imageTag, req.Data)

	f.logger.Debug("Function call completed",
		"functionID", functionID,
		"instanceID", instanceID,
		"predictedRuntime", prediction.Runtime,
		"responseSize", len(responseData),
	)

	return &common.CallResponse{
		Data: responseData,
	}, nil
}

// AddInstance adds a new instance to the router
func (f *FakeCallRouter) AddInstance(functionID string, instanceIP string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.functionInstances[functionID] = append(f.functionInstances[functionID], instanceIP)

	if f.activeCallCounts[functionID] == 0 {
		f.activeCallCounts[functionID] = 0
	}

	f.logger.Debug("Added instance to fake router",
		"functionID", functionID,
		"instanceIP", instanceIP,
		"totalInstances", len(f.functionInstances[functionID]),
	)
}

// HandleInstanceTimeout removes an instance from the router
func (f *FakeCallRouter) HandleInstanceTimeout(functionID string, instanceIP string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	instances := f.functionInstances[functionID]
	newInstances := make([]string, 0, len(instances))

	for _, ip := range instances {
		if ip != instanceIP {
			newInstances = append(newInstances, ip)
		}
	}

	f.functionInstances[functionID] = newInstances

	f.logger.Debug("Removed instance from fake router",
		"functionID", functionID,
		"instanceIP", instanceIP,
		"remainingInstances", len(newInstances),
	)
}

// HandleAddInstance is an alias for AddInstance for compatibility
func (f *FakeCallRouter) HandleAddInstance(functionID string, instanceIP string) {
	f.AddInstance(functionID, instanceIP)
}

// Helper functions

func extractInstanceIDFromIP(instanceIP string) string {
	// instanceIP format: "fake-instanceID:50052"
	if len(instanceIP) > 5 && instanceIP[:5] == "fake-" {
		// Find the colon and extract the instance ID
		for i := 5; i < len(instanceIP); i++ {
			if instanceIP[i] == ':' {
				return instanceIP[5:i]
			}
		}
	}
	return instanceIP // fallback
}

func (f *FakeCallRouter) getImageTagForInstance(instanceID string, containers map[string]*fakeRuntime.FakeContainer) string {
	if container, exists := containers[instanceID]; exists {
		return container.ImageTag
	}
	return "unknown" // fallback
}

func (f *FakeCallRouter) getCurrentWorkerCPU() float64 {
	// Simulate worker CPU usage based on active calls
	f.mu.RLock()
	activeCalls := f.globalActiveCallCount
	f.mu.RUnlock()

	// Simple heuristic: base CPU + calls * factor
	baseCPU := 0.1
	cpuPerCall := 0.05
	return baseCPU + float64(activeCalls)*cpuPerCall
}

func (f *FakeCallRouter) getCurrentWorkerRAM() int64 {
	// Simulate worker RAM usage based on running containers
	runningContainers := f.runtime.GetRunningContainers()

	// Simple heuristic: base RAM + containers * factor
	baseRAM := int64(512 * 1024 * 1024)         // 512MB base
	ramPerContainer := int64(100 * 1024 * 1024) // 100MB per container

	return baseRAM + int64(len(runningContainers))*ramPerContainer
}

func (f *FakeCallRouter) generateFakeResponse(imageTag string, requestData []byte) []byte {
	// Generate fake responses based on function type
	switch {
	case contains(imageTag, "hello"):
		return []byte("Hello, World!")
	case contains(imageTag, "echo"):
		return requestData // Echo back the request
	case contains(imageTag, "thumbnailer"):
		// Simulate thumbnail generation - return fake image data
		return []byte("fake-thumbnail-data-" + string(requestData))
	case contains(imageTag, "bfs"):
		// Simulate BFS result
		return []byte(`{"result": "fake-bfs-path", "steps": 42}`)
	case contains(imageTag, "sleep"):
		return []byte("slept successfully")
	default:
		return []byte("fake-response-" + string(requestData))
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) &&
			(s[:len(substr)] == substr ||
				s[len(s)-len(substr):] == substr ||
				indexOf(s, substr) >= 0)))
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
