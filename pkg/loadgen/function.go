package loadgen

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/3s-rg-codes/HyperFaaS/proto/common"
	"github.com/3s-rg-codes/HyperFaaS/proto/leaf"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type FunctionManager struct {
	conn      *grpc.ClientConn
	client    leaf.LeafClient
	functions map[string]*Function
}

type Function struct {
	ID          string
	ImageTag    string `yaml:"image_tag"`
	Timeout     int32  `yaml:"timeout"`
	ProtoConfig *common.Config
}

type FunctionConfig struct {
	Memory string            `yaml:"memory"`
	Cpu    *common.CPUConfig `yaml:"cpu"`
}

func NewFunctionManager(leafAddress string) *FunctionManager {
	conn, err := grpc.NewClient(leafAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Leaf: %v", err)
	}
	client := leaf.NewLeafClient(conn)
	return &FunctionManager{
		conn:      conn,
		client:    client,
		functions: make(map[string]*Function),
	}
}

func (f *FunctionManager) CreateFunction(imageTag string, timeout int32, functionConfig *FunctionConfig) *Function {

	// convert string memory to int64
	memory, err := convertMemory(functionConfig.Memory)
	if err != nil {
		log.Fatalf("failed to convert memory: %v", err)
	}

	protoConfig := &common.Config{
		Memory: memory,
		Cpu:    functionConfig.Cpu,
	}

	r, err := f.client.CreateFunction(context.Background(), &leaf.CreateFunctionRequest{
		ImageTag: &common.ImageTag{
			Tag: imageTag,
		},
		Config: protoConfig,
	})
	if err != nil {
		log.Fatalf("Failed to create function: %v", err)
	}

	return &Function{
		ID:          r.FunctionID.Id,
		ImageTag:    imageTag,
		Timeout:     timeout,
		ProtoConfig: protoConfig,
	}
}

func convertMemory(memory string) (int64, error) {
	memory = strings.ToUpper(strings.TrimSpace(memory))

	var multiplier int64
	var numStr string

	if strings.HasSuffix(memory, "GB") {
		multiplier = 1024 * 1024 * 1024 // 1 GB in bytes
		numStr = strings.TrimSuffix(memory, "GB")
	} else if strings.HasSuffix(memory, "MB") {
		multiplier = 1024 * 1024 // 1 MB in bytes
		numStr = strings.TrimSuffix(memory, "MB")
	} else {
		return 0, fmt.Errorf("unsupported memory format: %s (expected format like '256MB' or '1GB')", memory)
	}

	num, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid memory value: %s", memory)
	}

	totalBytes := int64(num * float64(multiplier))

	minBytes := int64(6 * 1024 * 1024)
	if totalBytes < minBytes {
		return 0, fmt.Errorf("memory must be at least 6MB, got %s", memory)
	}

	return totalBytes, nil
}
