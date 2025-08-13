package loadgen

import (
	"context"
	"log"
	"time"

	"github.com/3s-rg-codes/HyperFaaS/proto/leaf"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type LeafClient struct {
	conn   *grpc.ClientConn
	client leaf.LeafClient
}

func NewLeafClient(address string) *LeafClient {
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Leaf: %v", err)
	}
	return &LeafClient{
		conn:   conn,
		client: leaf.NewLeafClient(conn),
	}
}

func (lc *LeafClient) Close() error {
	return lc.conn.Close()
}

func (lc *LeafClient) ScheduleCall(ctx context.Context, req *leaf.ScheduleCallRequest) (CallResult, error) {
	var trailers metadata.MD
	var headers metadata.MD

	start := time.Now()
	resp, err := lc.client.ScheduleCall(ctx, req,
		grpc.Header(&headers),
		grpc.Trailer(&trailers))
	latency := time.Since(start)

	if err != nil {
		return CallResult{
			Timestamp:    start,
			FunctionID:   req.FunctionID.Id,
			Latency:      latency,
			Status:       status.Code(err),
			Error:        err.Error(),
			ResponseSize: 0,
		}, err
	}

	// Extract HyperFaaS-specific trailer values
	result := CallResult{
		Timestamp:                  start,
		FunctionID:                 req.FunctionID.Id,
		Latency:                    latency,
		Status:                     status.Code(err),
		ResponseSize:               int64(len(resp.Data)),
		CallQueuedTimestamp:        getTrailerValue(trailers, "callQueuedTimestamp"),
		GotResponseTimestamp:       getTrailerValue(trailers, "gotResponseTimestamp"),
		InstanceID:                 getTrailerValue(trailers, "instanceId"),
		LeafGotRequestTimestamp:    getTrailerValue(trailers, "leafGotRequestTimestamp"),
		LeafScheduledCallTimestamp: getTrailerValue(trailers, "leafScheduledCallTimestamp"),
		FunctionProcessingTime:     getTrailerValue(trailers, "functionProcessingTime"),
	}

	return result, err
}

func getTrailerValue(md metadata.MD, key string) string {
	values := md.Get(key)
	if len(values) > 0 {
		return values[0]
	}
	log.Println("No trailer value found for key:", key)
	return ""
}
