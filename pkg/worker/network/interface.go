package network

import "github.com/3s-rg-codes/HyperFaaS/proto/common"

// CallRouterInterface defines the interface that both real and fake routers must implement
type CallRouterInterface interface {
	// CallFunction executes a function call and returns the response
	CallFunction(functionID string, req *common.CallRequest) (*common.CallResponse, error)

	// AddInstance adds a new function instance to the router
	AddInstance(functionID string, instanceIP string)

	// HandleInstanceTimeout removes an instance that has timed out
	HandleInstanceTimeout(functionID string, instanceIP string)
}
