package loadgen

import (
	"context"
	"log/slog"
	"time"

	"github.com/3s-rg-codes/HyperFaaS/proto/common"
	"github.com/3s-rg-codes/HyperFaaS/proto/leaf"
)

type LoadExecutor interface {
	Execute(ctx context.Context, phase TestPhase) error
	Stop()
}

type client interface {
	ScheduleCall(ctx context.Context, req *leaf.ScheduleCallRequest) (CallResult, error)
}

type dataCollector interface {
	Collect(result CallResult)
}

type ConstantExecutor struct {
	rps          int
	client       client
	collector    dataCollector
	funcMgr      *FunctionManager
	l            *slog.Logger
	dataProvider DataProvider
}

func NewConstantExecutor(client client, collector dataCollector, funcMgr *FunctionManager, l *slog.Logger, dataProvider DataProvider) *ConstantExecutor {
	return &ConstantExecutor{
		client:       client,
		collector:    collector,
		funcMgr:      funcMgr,
		l:            l,
		dataProvider: dataProvider,
	}
}

type RampingExecutor struct {
	startRPS     int
	endRPS       int
	step         int
	duration     time.Duration
	client       client
	collector    dataCollector
	funcMgr      *FunctionManager
	l            *slog.Logger
	dataProvider DataProvider
}

func NewRampingExecutor(client client, collector dataCollector, funcMgr *FunctionManager, l *slog.Logger, dataProvider DataProvider) *RampingExecutor {
	return &RampingExecutor{
		client:       client,
		collector:    collector,
		funcMgr:      funcMgr,
		l:            l,
		dataProvider: dataProvider,
	}
}

func (e *ConstantExecutor) Execute(ctx context.Context, phase TestPhase) {
	e.rps = phase.StartRPS

	subCtx, cancel := context.WithTimeout(ctx, phase.Duration)
	defer cancel()

	t := time.NewTicker(time.Second)

	for {
		select {
		case <-subCtx.Done():
			return
		case <-t.C:
			e.l.Debug("Constant executor", "Current RPS", e.rps)
			for i := 0; i < e.rps; i++ {

				go func() {
					data := e.dataProvider.GetData()
					result, _ := e.client.ScheduleCall(ctx, &leaf.ScheduleCallRequest{
						FunctionID: &common.FunctionID{
							Id: phase.FunctionID,
						},
						Data: data,
					})
					result.ImageTag = phase.ImageTag
					result.RequestSize = int64(len(data))
					e.collector.Collect(result)
				}()

			}
		}
	}
}

func (e *RampingExecutor) Execute(ctx context.Context, phase TestPhase) {
	e.startRPS = phase.StartRPS
	e.endRPS = phase.EndRPS
	e.step = phase.Step
	incrementing := e.step > 0

	subCtx, cancel := context.WithTimeout(ctx, phase.Duration)
	defer cancel()

	currentRPS := e.startRPS
	if currentRPS == 0 {
		e.l.Warn("Start RPS is 0, setting to 1")
		currentRPS = 1
	}

	t := time.NewTicker(time.Second)
	first := true
	for {
		select {
		case <-subCtx.Done():
			return
		case <-t.C:
			e.l.Debug("Ramping executor", "Current RPS", currentRPS)
			if !first && (incrementing && currentRPS < e.endRPS || !incrementing && currentRPS > e.endRPS) {
				currentRPS += e.step
			}

			if currentRPS <= 0 {
				break
			}
			first = false

			for i := 0; i < currentRPS; i++ {
				go func() {
					data := e.dataProvider.GetData()
					result, _ := e.client.ScheduleCall(ctx, &leaf.ScheduleCallRequest{
						FunctionID: &common.FunctionID{
							Id: phase.FunctionID,
						},
						Data: data,
					})
					result.ImageTag = phase.ImageTag
					result.RequestSize = int64(len(data))
					e.collector.Collect(result)
				}()
			}
		}
	}
}
