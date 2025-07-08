package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"time"

	"github.com/3s-rg-codes/HyperFaaS/pkg/keyValueStore"

	"github.com/3s-rg-codes/HyperFaaS/pkg/leaf/api"
	"github.com/3s-rg-codes/HyperFaaS/pkg/leaf/config"
	"github.com/3s-rg-codes/HyperFaaS/pkg/leaf/state"
	pb "github.com/3s-rg-codes/HyperFaaS/proto/leaf"
	"github.com/golang-cz/devslog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

type workerIDs []string

func (i *workerIDs) String() string {
	return fmt.Sprintf("%v", *i)
}

func (i *workerIDs) Set(value string) error {
	*i = append(*i, value)
	return nil
}

func main() {
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()
	workerIDs := workerIDs{}
	address := flag.String("address", "0.0.0.0:50050", "The address to listen on")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error) (Env: LOG_LEVEL)")
	logFormat := flag.String("log-format", "text", "Log format (json, text or dev) (Env: LOG_FORMAT)")
	logFilePath := flag.String("log-file", "", "Log file path (defaults to stdout) (Env: LOG_FILE)")
	databaseType := flag.String("database-type", "http", "\"database\" used for managing the functionID -> config relationship")
	databaseAddress := flag.String("database-address", "http://localhost:8999/", "address of the database server")
	maxStartingInstancesPerFunction := flag.Int("max-starting-instances-per-function", 10, "The maximum number of instances starting at once per function")
	startingInstanceWaitTimeout := flag.Duration("starting-instance-wait-timeout", time.Second*5, "The timeout for waiting for an instance to start")
	maxRunningInstancesPerFunction := flag.Int("max-running-instances-per-function", 10, "The maximum number of instances running at once per function")
	panicBackoff := flag.Duration("panic-backoff", time.Millisecond*50, "The starting backoff time for the panic mode")
	panicBackoffIncrease := flag.Duration("panic-backoff-increase", time.Millisecond*50, "The backoff increase for the panic mode")
	panicMaxBackoff := flag.Duration("panic-max-backoff", time.Second*1, "The maximum backoff for the panic mode")

	// Memory optimization flags
	maxConnectionIdle := flag.Duration("max-connection-idle", 15*time.Second, "Maximum time a connection can be idle before being closed")
	maxConnectionAge := flag.Duration("max-connection-age", 30*time.Second, "Maximum age of a connection before being gracefully closed")
	maxConnectionAgeGrace := flag.Duration("max-connection-age-grace", 5*time.Second, "Grace period for closing aged connections")
	keepaliveTime := flag.Duration("keepalive-time", 5*time.Second, "Time between keepalive pings")
	keepaliveTimeout := flag.Duration("keepalive-timeout", 1*time.Second, "Timeout for keepalive pings")
	keepaliveMinTime := flag.Duration("keepalive-min-time", 1*time.Second, "Minimum time between keepalive pings")
	enableSharedWriteBuffer := flag.Bool("enable-shared-write-buffer", true, "Enable shared write buffer for memory efficiency")

	flag.Var(&workerIDs, "worker-ids", "The IDs of the workers to manage")
	flag.Parse()

	if len(workerIDs) == 0 {
		panic("no worker IDs provided")
	}

	logger := setupLogger(*logLevel, *logFormat, *logFilePath)

	// Print configuration
	logger.Info("Configuration", "address", *address, "logLevel", *logLevel, "logFormat", *logFormat, "logFilePath", *logFilePath, "databaseType", *databaseType, "databaseAddress", *databaseAddress, "workerIDs", workerIDs)
	logger.Info("Memory Optimization Settings", "maxConnectionIdle", *maxConnectionIdle, "maxConnectionAge", *maxConnectionAge, "maxConnectionAgeGrace", *maxConnectionAgeGrace, "keepaliveTime", *keepaliveTime, "keepaliveTimeout", *keepaliveTimeout, "keepaliveMinTime", *keepaliveMinTime, "enableSharedWriteBuffer", *enableSharedWriteBuffer)

	var ids []state.WorkerID
	logger.Debug("Setting worker IDs", "workerIDs", workerIDs, "len", len(workerIDs))
	for _, id := range workerIDs {
		err := healthCheckWorker(id)
		if err != nil {
			logger.Error("failed to health check worker", "error", err)
			os.Exit(1)
		}
		ids = append(ids, state.WorkerID(id))
	}

	var dbClient keyValueStore.FunctionMetadataStore

	switch *databaseType {
	case "http":
		dbClient = keyValueStore.NewHttpClient(*databaseAddress, logger)
	}

	leafConfig := config.LeafConfig{
		MaxStartingInstancesPerFunction: *maxStartingInstancesPerFunction,
		StartingInstanceWaitTimeout:     *startingInstanceWaitTimeout,
		MaxRunningInstancesPerFunction:  *maxRunningInstancesPerFunction,
		PanicBackoff:                    *panicBackoff,
		PanicBackoffIncrease:            *panicBackoffIncrease,
		PanicMaxBackoff:                 *panicMaxBackoff,
	}

	server := api.NewLeafServer(leafConfig, dbClient, ids, logger)

	listener, err := net.Listen("tcp", *address)
	if err != nil {
		logger.Error("failed to listen", "error", err)
		os.Exit(1)
	}

	// Configure gRPC server with memory optimizations for handling many connections
	keepaliveParams := keepalive.ServerParameters{
		MaxConnectionIdle:     *maxConnectionIdle,
		MaxConnectionAge:      *maxConnectionAge,
		MaxConnectionAgeGrace: *maxConnectionAgeGrace,
		Time:                  *keepaliveTime,
		Timeout:               *keepaliveTimeout,
	}

	keepalivePolicy := keepalive.EnforcementPolicy{
		MinTime:             *keepaliveMinTime,
		PermitWithoutStream: true, // Allow pings even when no streams are active
	}

	var grpcOptions []grpc.ServerOption
	grpcOptions = append(grpcOptions, grpc.KeepaliveParams(keepaliveParams))
	grpcOptions = append(grpcOptions, grpc.KeepaliveEnforcementPolicy(keepalivePolicy))

	if *enableSharedWriteBuffer {
		grpcOptions = append(grpcOptions, grpc.SharedWriteBuffer(true)) // Enable shared write buffer for better memory efficiency, should help with memory usage given the shtty load generator creating so many connections
	}

	grpcServer := grpc.NewServer(grpcOptions...)
	pb.RegisterLeafServer(grpcServer, server)
	logger.Info("Leaf server started with memory optimizations", "address", listener.Addr(),
		"maxConnectionIdle", keepaliveParams.MaxConnectionIdle,
		"maxConnectionAge", keepaliveParams.MaxConnectionAge,
		"keepaliveTime", keepaliveParams.Time,
		"sharedWriteBuffer", *enableSharedWriteBuffer)
	if err := grpcServer.Serve(listener); err != nil {
		logger.Error("failed to serve", "error", err)
		os.Exit(1)
	}

}

func setupLogger(logLevel string, logFormat string, logFilePath string) *slog.Logger {
	// Set up log level
	var level slog.Level
	switch logLevel {
	case "debug":
		level = slog.LevelDebug
	case "info":
		level = slog.LevelInfo
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	// Set up log output
	var output *os.File
	var err error
	if logFilePath != "" {
		output, err = os.OpenFile(logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			slog.Error("Failed to open log file, falling back to stdout", "error", err)
			output = os.Stdout
		}
	} else {
		output = os.Stdout
	}

	// Set up handler options
	opts := &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
	}

	// Create handler based on format
	var handler slog.Handler
	switch logFormat {
	case "json":
		handler = slog.NewJSONHandler(output, opts)
	case "text":
		handler = slog.NewTextHandler(output, opts)
	case "dev":
		devOpts := &devslog.Options{
			HandlerOptions:    opts,
			MaxSlicePrintSize: 5,
			SortKeys:          true,
			NewLineAfterLog:   true,
			StringerFormatter: true,
		}
		handler = devslog.NewHandler(output, devOpts)
	}
	logger := slog.New(handler)
	slog.SetDefault(logger)

	return logger
}

var serviceConfig = `{
	"loadBalancingPolicy": "round_robin",
	"healthCheckConfig": {
		"serviceName": ""
	}
}`

func healthCheckWorker(workerID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// This function is deprecated but I'm not sure how to replace it
	//https://github.com/grpc/grpc-go/blob/master/Documentation/anti-patterns.md
	conn, err := grpc.DialContext(ctx, workerID,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(), // This makes the dial synchronous
	)
	if err != nil {
		return fmt.Errorf("failed to connect to worker %s: %w", workerID, err)
	}
	defer conn.Close()

	return nil
}
