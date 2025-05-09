package state

import "fmt"

type FunctionNotAssignedError struct {
	FunctionID FunctionID
}

func (e *FunctionNotAssignedError) Error() string {
	return fmt.Sprintf("function %s not registered with any worker", e.FunctionID)
}

type NoIdleInstanceError struct {
	FunctionID FunctionID
}

func (e *NoIdleInstanceError) Error() string {
	return fmt.Sprintf("no idle instance found for function %s", e.FunctionID)
}

type WorkerNotFoundError struct {
	WorkerID WorkerID
}

func (e *WorkerNotFoundError) Error() string {
	return fmt.Sprintf("worker %s not found", e.WorkerID)
}
