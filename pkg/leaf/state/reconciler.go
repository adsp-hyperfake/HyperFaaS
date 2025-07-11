package state

import (
	"context"
	"io"
	"time"

	controllerpb "github.com/3s-rg-codes/HyperFaaS/proto/controller"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	// If multiple leaf nodes listen to the same worker, each needs a different leafNodeID
	leafNodeID = "todo"
)

// The reconciler is responsible for reconciling the state of the workers and instances.
// It reads the StatusUpdate stream from the workers and updates the state of the scheduler if necessary.
// The most common case is that instances time out when waiting for more calls.
/* type Reconciler struct {
	workerIDs []WorkerID
	workers   WorkerData
	logger    *slog.Logger
}

// We accept any type that implements RemoveInstance
type WorkerData interface {
	RemoveInstance(workerID WorkerID, functionID FunctionID, instanceID InstanceID) error
}

func NewReconciler(workerIDs []WorkerID, workers WorkerData, logger *slog.Logger) *Reconciler {
	return &Reconciler{
		workerIDs: workerIDs,
		workers:   workers,
		logger:    logger,
	}
} */

// RunReconciler runs the reconciler asynchronously in a loop, listening to the status updates from the workers.
// It is responsible for updating the state if necessary, for example if a container times out or is down.
func (s *SmallState) RunReconciler(ctx context.Context) {
	for _, workerID := range s.workers {
		go s.ListenToWorkerStatusUpdates(ctx, workerID)
	}
}

func (s *SmallState) getStatusUpdateStream(ctx context.Context, workerID WorkerID) (controllerpb.Controller_StatusClient, *grpc.ClientConn, error) {

	conn, err := grpc.NewClient(string(workerID), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		s.logger.Error("Failed to create gRPC client", "error", err)
		return nil, nil, err
	}

	client := controllerpb.NewControllerClient(conn)

	statusUpdates, err := client.Status(ctx, &controllerpb.StatusRequest{NodeID: leafNodeID})
	if err != nil {
		s.logger.Error("Failed to get status updates", "error", err)
		return nil, nil, err
	}

	return statusUpdates, conn, nil

}

func (s *SmallState) ListenToWorkerStatusUpdates(ctx context.Context, workerID WorkerID) {
	for {
		statusUpdates, conn, err := s.getStatusUpdateStream(ctx, workerID)
		if err != nil {
			s.logger.Error("Failed to get status update stream", "workerID", workerID, "error", err)
			// Wait a bit before retrying
			time.Sleep(5 * time.Second)
			continue
		}

		func() {
			defer conn.Close()
			for {
				update, err := statusUpdates.Recv()
				if err == io.EOF {
					s.logger.Debug("Status update stream closed", "workerID", workerID)
					return
				}
				if err != nil {
					s.logger.Error("Failed to receive status update", "error", err)
					return
				}

				switch update.Type {
				case controllerpb.VirtualizationType_TYPE_CONTAINER:
					switch update.Event {
					case controllerpb.Event_EVENT_TIMEOUT:
						s.handleContainerTimeout(workerID, FunctionID(update.FunctionId.Id), InstanceID(update.InstanceId.Id))
					case controllerpb.Event_EVENT_DOWN:
						s.handleContainerDown(workerID, FunctionID(update.FunctionId.Id), InstanceID(update.InstanceId.Id))
					case controllerpb.Event_EVENT_START:
					case controllerpb.Event_EVENT_STOP:
					case controllerpb.Event_EVENT_RUNNING:
					case controllerpb.Event_EVENT_CALL:
					default:
						//r.logger.Warn("Received status update of unknown event", "event", update.Event)
					}
				default:
					s.logger.Warn("Received status update of unknown type", "type", update.Type)
				}
			}
		}()
	}
}

func (s *SmallState) handleContainerTimeout(workerID WorkerID, functionID FunctionID, instanceID InstanceID) {
	s.logger.Debug("Container timed out", "instanceID", instanceID)

	autoscaler, ok := s.GetAutoscaler(functionID)
	if !ok {
		s.logger.Error("Reconciliation failed to find autoscaler", "functionID", functionID)
		return
	}
	autoscaler.UpdateRunningInstances(-1)
}

func (s *SmallState) handleContainerDown(workerID WorkerID, functionID FunctionID, instanceID InstanceID) {
	s.logger.Debug("Container down", "instanceID", instanceID)

	autoscaler, ok := s.GetAutoscaler(functionID)
	if !ok {
		s.logger.Error("Reconciliation failed to find autoscaler", "functionID", functionID)
		return
	}
	autoscaler.UpdateRunningInstances(-1)
}
