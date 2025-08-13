package loadgen

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
)

var (
	CSV_HEADERS = []string{"timestamp", "function_id", "image_tag", "latency_ms", "status", "error", "request_size_bytes", "response_size_bytes", "call_queued_timestamp", "got_response_timestamp", "instance_id", "leaf_got_request_timestamp", "leaf_scheduled_call_timestamp", "function_processing_time_ns"}
)

type Collector struct {
	csvWriter *csv.Writer
	fileName  string
	file      *os.File
	mutex     sync.Mutex
	headers   []string
}

func NewCollector(fileName string) *Collector {
	file, err := os.Create(fileName)
	if err != nil {
		log.Fatalf("Failed to create results file: %v", err)
	}
	writer := csv.NewWriter(file)
	writer.Write(CSV_HEADERS)
	writer.Flush()
	return &Collector{
		csvWriter: writer,
		fileName:  fileName,
		file:      file,
		headers:   CSV_HEADERS,
	}
}

type CallResult struct {
	Timestamp time.Time

	Latency      time.Duration
	Status       codes.Code
	Error        string
	RequestSize  int64
	ResponseSize int64
	// HyperFaaS-specific trailer fields
	FunctionID                 string
	ImageTag                   string
	CallQueuedTimestamp        string
	GotResponseTimestamp       string
	InstanceID                 string
	LeafGotRequestTimestamp    string
	LeafScheduledCallTimestamp string
	FunctionProcessingTime     string
}

func (c *Collector) Collect(result CallResult) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.csvWriter.Write([]string{
		result.Timestamp.Format(time.RFC3339),
		result.FunctionID,
		result.ImageTag,
		strconv.FormatInt(result.Latency.Nanoseconds(), 10),
		result.Status.String(),
		result.Error,
		strconv.FormatInt(result.RequestSize, 10),
		strconv.FormatInt(result.ResponseSize, 10),
		result.CallQueuedTimestamp,
		result.GotResponseTimestamp,
		result.InstanceID,
		result.LeafGotRequestTimestamp,
		result.LeafScheduledCallTimestamp,
		result.FunctionProcessingTime,
	})
}

// TODO: make this configurable
func (c *Collector) RunFlusher() {
	t := time.NewTicker(time.Second)
	defer t.Stop()

	for range t.C {
		c.csvWriter.Flush()
	}
}

func (c *Collector) Close() {
	c.csvWriter.Flush()
	if c.file != nil {
		c.file.Close()
	}
}
