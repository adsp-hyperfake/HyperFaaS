package fake

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	fakeRuntime "github.com/3s-rg-codes/HyperFaaS/pkg/worker/containerRuntime/fake"
	"github.com/3s-rg-codes/HyperFaaS/proto/common"
)

// FakeCallRouter simulates function calls using mathematical models
type FakeCallRouter struct {
	mu                sync.RWMutex
	functionInstances map[string][]string // functionID -> list of instanceIPs
	runtime           *fakeRuntime.FakeContainerRuntime
	logger            *slog.Logger
}

// NewFakeCallRouter creates a new fake call router
func NewFakeCallRouter(runtime *fakeRuntime.FakeContainerRuntime, logger *slog.Logger) *FakeCallRouter {
	return &FakeCallRouter{
		functionInstances: make(map[string][]string),
		runtime:           runtime,
		logger:            logger,
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

	return f.runtime.Call(context.Background(), req)
}

// AddInstance adds a new instance to the router
func (f *FakeCallRouter) AddInstance(functionID string, instanceIP string) {
	f.mu.Lock()
	f.functionInstances[functionID] = append(f.functionInstances[functionID], instanceIP)
	f.mu.Unlock()

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
