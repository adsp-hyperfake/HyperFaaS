package main

import (
	"flag"
	"log/slog"
	"net/http"
	_ "net/http/pprof"
	"os"

	"github.com/3s-rg-codes/HyperFaaS/pkg/loadgen"
)

func main() {
	go func() {
		http.ListenAndServe("localhost:6060", nil)
	}()
	config := flag.String("config", "workload_config.yaml", "config file")
	out := flag.String("out", "results.csv", "output collector file name")
	logLevel := flag.String("log-level", "info", "log level")
	flag.Parse()

	logLevelInt := getLogLevel(*logLevel)

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.Level(logLevelInt),
	}))

	controller := loadgen.NewController(
		logger,
		loadgen.WithConfigFile(*config),
		loadgen.WithCollector(loadgen.NewCollector(*out)),
	)

	controller.Run()
}

func getLogLevel(logLevel string) slog.Level {
	switch logLevel {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
