# For more information, see https://github.com/casey/just

set dotenv-load
set export
set windows-shell := ["powershell"]
set shell := ["bash", "-c"]

default:
  @just --list --unsorted

############################
# Building Stuff
############################

# generate the proto files
proto:
    @echo "Generating proto files"
    protoc --proto_path=proto "proto/function/function.proto" --go_out=proto --go_opt=paths=source_relative --go-grpc_out=proto --go-grpc_opt=paths=source_relative
    protoc --proto_path=proto "proto/controller/controller.proto" --go_out=proto --go_opt=paths=source_relative --go-grpc_out=proto --go-grpc_opt=paths=source_relative
    protoc --proto_path=proto "proto/leaf/leaf.proto" --go_out=proto --go_opt=paths=source_relative --go-grpc_out=proto --go-grpc_opt=paths=source_relative
    protoc --proto_path=proto "proto/common/common.proto" --go_out=proto --go_opt=paths=source_relative --go-grpc_out=proto --go-grpc_opt=paths=source_relative

# build the worker binary
build-worker:
    go build -o bin/ cmd/worker/main.go
leaf:
    docker compose up -d --build leaf
worker:
    docker compose up -d --build worker
# build all go functions
build-functions-go:
    find ./functions/go/*/ -maxdepth 0 -type d | xargs -I {} bash -c 'just build-function-go $(basename "{}")'


# build a single go function
build-function-go function_name:
    # Pro Tip: you can add `--progress=plain` as argument to build, and then the full output of your build will be shown.
    docker build -t hyperfaas-{{function_name}} -f ./function.Dockerfile --build-arg FUNCTION_NAME="{{function_name}}" .

# build all
build: build-functions-go build-worker




############################
# Running Stuff
############################

# run the worker with default configurations. Make sure to run just build every time you change the code
# Alternatively, run just dev if you want to make sure you are always running the latest code
start-rebuild:
    @echo "Starting docker service"
    docker compose up --build

start:
    @echo "Starting docker service"
    WORKER_TYPE=worker docker compose up --scale fake-worker=0 --detach --remove-orphans

restart:
    @echo "Restarting docker service"
    WORKER_TYPE=worker docker compose restart

stop:
    @echo "Stopping docker service"
    WORKER_TYPE=worker docker compose down

d:
    @echo "Starting docker service"
    WORKER_TYPE=worker docker compose up --scale fake-worker=0 --build --detach

############################
# Running Faked HyperFaaS
############################

# make sure that the onnx models are in hyperFakeModel/
fake-start runtime_type:
    FAKE_RUNTIME_TYPE={{runtime_type}} WORKER_TYPE=fake-worker docker compose up --scale worker=0 fake-worker leaf database -d --build
fake-stop:
    WORKER_TYPE=fake-worker docker compose down
fake-restart:
    WORKER_TYPE=fake-worker docker compose restart

# generates proto, builds binary, builds docker go and runs the workser
dev: build start

run-local-database:
    @echo "Running local database"
    go run cmd/database/main.go --address=0.0.0.0:8999

run-local-worker:
    @echo "Running local worker"
    go run cmd/worker/main.go --address=localhost:50051 --runtime=docker --log-level=info --log-format=dev --auto-remove=true --containerized=false --caller-server-address=127.0.0.1:50052 --database-type=http
run-local-fake-worker:
    @echo "Running local fake worker"
    just hyperFakeWorker/run-local

run-local-leaf:
    @echo "Running local leaf"
    go run cmd/leaf/main.go --address=localhost:50050 --log-level=debug --log-format=text --worker-ids=127.0.0.1:50051 --database-address=http://localhost:8999

################################
# Training Stuff - Random Forest
################################

train-random-forest db_path="../../benchmarks/metrics.db" table="training_data" n_estimators="100" max_depth="" min_samples_split="2" min_samples_leaf="1" max_features="sqrt" random_state="42" n_jobs="-1":
    #!/bin/bash
    cd hyperFakeModel/random-forest
    uv sync
    
    # Build the command with conditional parameters
    cmd="uv run random_forest.py --db-path {{db_path}} --table {{table}} --n-estimators {{n_estimators}} --min-samples-split {{min_samples_split}} --min-samples-leaf {{min_samples_leaf}} --max-features {{max_features}} --random-state {{random_state}} --n-jobs {{n_jobs}}"
    
    # Add max-depth only if specified (since None is the default)
    if [ "{{max_depth}}" != "" ]; then
        cmd="$cmd --max-depth {{max_depth}}"
    fi
    
    echo "Running: $cmd"
    eval $cmd

train-ridge-regression db_path="../../benchmarks/metrics.db" table="training_data" alpha_min="-3" alpha_max="3" alpha_num="25" cv_folds="5" test_size="0.2" val_size="0.25" random_state="42":
    cd hyperFakeModel/ridge-regression && \
        uv sync && \
        uv run ridge_regression.py --db-path {{db_path}} --table {{table}} --alpha-min {{alpha_min}} --alpha-max {{alpha_max}} --alpha-num {{alpha_num}} --cv-folds {{cv_folds}} --test-size {{test_size}} --val-size {{val_size}} --random-state {{random_state}}


#################################
# Training Stuff - Neural Network
#################################

NEURAL_VENV_PATH := "./neural-net-venv"
NEURAL_PYTHON := NEURAL_VENV_PATH + "/bin/python"
NEURAL_PIP := NEURAL_VENV_PATH + "/bin/pip"
NEURAL_PATH := "./hyperFakeModel/neural-network"
NEURAL_CLI := NEURAL_PATH + "/neural_net_cli.py"

# Set up the venv
neural-setup-venv python_version="python3.12":
    rm -rf {{NEURAL_VENV_PATH}}
    {{python_version}} -m venv {{NEURAL_VENV_PATH}} --prompt "neural-net-venv"
    {{NEURAL_PIP}} install --upgrade pip
    {{NEURAL_PIP}} install -r {{NEURAL_PATH}}/requirements.txt

# Run Optuna optimization to get hyperparameters
neural-optuna function trials="150" epochs="50" jobs="-1" final-epochs="0" samples="-1" extra_args="":
    {{NEURAL_PYTHON}} {{NEURAL_CLI}} optuna --func-tag hyperfaas-{{function}}:latest --short-name {{function}} \
        --trials {{trials}} \
        --epochs {{epochs}} \
        --jobs {{jobs}} \
        --final-epochs {{final-epochs}} \
        --samples {{samples}} \
        {{extra_args}}

# Test the above optimization with a quick run, cleans up automatically
neural-optuna-test function:
    #!/bin/bash
    echo "Creating temporary dir"
    mkdir ./tmpoptunatest
    just neural-optuna {{function}} 5 5 2 5 100 "--export-dir ./tmpoptunatest"
    printf "\n\n"
    for f in ./tmpoptunatest/*; do
        echo "File $f written"
        if [[ $f == *.json ]]; then
            echo "hyperparams.json:"
            cat "$f"
        fi
    done
    rm -rf ./tmpoptunatest
    echo "\nTemporary dir removed"

# Train the actual model
neural-train-model function epochs="200" hyperparams_path="" extra_args="":
    #!/bin/bash
    # if no path is provided, look for the hyperparams in the default directory
    if [ -z "{{hyperparams_path}}" ]; then
        matches=(${NEURAL_PATH}/models/${function}_*_hyperparams.json)
        if [ "${matches[0]}" = "./tmpoptunatest/{{function}}_*_hyperparams.json" ]; then
            echo "No hyperparams file found for {{function}}"
            exit 1
        fi
        if [ "${#matches[@]}" -gt 1 ]; then
            echo "More than one hyperparams file found for {{function}}, please specify which one to use:"
            for f in "${matches[@]}"; do
                echo "  $f"
            done
            exit 1
        fi
        # Exactly one match
        hyperparams_path="${matches[0]}"
        echo "No hyperparams specified, continuing with hyperparams found at $hyperparams_path"
    fi
    {{NEURAL_PYTHON}} {{NEURAL_CLI}} manual --func-tag hyperfaas-{{function}}:latest --short-name {{function}} --epochs {{epochs}} --hyperparams $hyperparams_path {{extra_args}}

# Train the model on the specified input columns (space-separated single argument) only
neural-train-model-cols function cols epochs="200" hyperparams_path="" extra_args="":
    #!/bin/bash
    args=""
    for w in $cols; do
        args+="--input-cols $w "
    done
    args+={{extra_args}}
    just neural-train-model {{function}} {{epochs}} "{{hyperparams_path}}" "$args"


# Move all contents of the ./hyperFakeModel/neural-network/models folder to a subfolder
neural-clean:
    #!/bin/bash
    timestamp=$(date +"%Y%m%d_%H%M%S")
    origin="./hyperFakeModel/neural-network/models/"
    target="./hyperFakeModel/neural-network/models/$timestamp"
    # Create the subfolder inside 'your_folder'
    mkdir -p $target
    # Move all contents from 'your_folder' (except the new subfolder) to the timestamp subfolder
    find $origin -mindepth 1 -maxdepth 1 -type f ! -name ".*" -exec mv {} $target \;

neural-copy-models origin="./hyperFakeModel/neural-network/models/" target="./hyperFakeModel/":
    find "{{origin}}" -mindepth 1 -maxdepth 1 -type f \( -name "*.onnx" -o -name "*.onnx.data" \) -exec cp {} "{{target}}" \; -exec echo "Copied: {}" \;

############################
# Testing Stuff
############################

# run the tests
test-all:
    go test -v ./...

# run a test with a specific test name
test name:
    go test -run {{name}} ./...

#Containerized integration tests via docker compose
build-integration-containerized-all:
    ENTRYPOINT_CMD="-test_cases=all" docker compose -f test-compose.yaml up --build

build-integration-containerized list:
    ENTRYPOINT_CMD="-test_cases={{list}}" docker compose -f test-compose.yaml up --build

test-integration-containerized-all:
    ENTRYPOINT_CMD="-test_cases=all" docker compose -f test-compose.yaml up

test-integration-containerized list:
    ENTRYPOINT_CMD="-test_cases={{list}}" docker compose -f test-compose.yaml up

#Local integration tests
test-integration-local-all runtime loglevel:
    cd ./cmd/database && go run . &
    cd ./tests/worker && go run . {{runtime}} {{loglevel}}

###### Metrics Tests ########
metrics-client:
    go run ./cmd/metrics-client

load-test:
    go run ./tests/leaf/main.go

metrics-analyse:
    cd benchmarks && uv run main.py --analyse --db-path metrics.db --plot-save-path ./plots/

metrics-plot:
    cd benchmarks && uv run plot.py--plot --prefix $(date +%Y-%m-%d) --plot-save-path ./plots/
metrics-process:
    cd benchmarks && uv run process.py --db-path ../benchmarks/metrics.db --active-calls-window-size 100

metrics-clean-training:
    sqlite3 benchmarks/metrics.db "drop table training_data;"
metrics-verify:
    sqlite3 benchmarks/metrics.db ".headers on" "select count(), function_instances_count from training_data group by function_instances_count limit 100;"
    sqlite3 benchmarks/metrics.db ".headers on" "select count() , active_function_calls_count from training_data group by active_function_calls_count limit 100;"
    sqlite3 benchmarks/metrics.db ".headers on" "select count(distinct(function_cpu_usage)) from training_data;"
    sqlite3 benchmarks/metrics.db ".headers on" "select count(distinct(function_ram_usage)) from training_data;"
    sqlite3 benchmarks/metrics.db ".headers on" "select count(distinct(worker_cpu_usage)) from training_data;"
    sqlite3 benchmarks/metrics.db ".headers on" "select count(distinct(worker_ram_usage)) from training_data;"
    sqlite3 benchmarks/metrics.db ".headers on" "select count(case when function_cpu_usage = 0.0 then 1 end) as zero_count, count(case when function_cpu_usage != 0.0 then 1 end) as non_zero_count from training_data;"
    sqlite3 benchmarks/metrics.db ".headers on" "select scenario, count() from metrics where grpc_req_duration is null and error is null and timeout is null group by scenario;"

############################
# Data pipeline
############################
run-full-pipeline config_file="benchmarks/configs/10m.yaml" out_file="results.csv":
    #!/bin/bash
    # run the load generation
    go run cmd/load-generator/main.go --config {{config_file}} --out benchmarks/{{out_file}}
    # call pull metrics script : this will fail unless you configure it correctly
    ./pull-metrics.sh
    # process the metrics
    cd benchmarks && uv run new_import.py --csv {{out_file}} --db metrics.db

    cd benchmarks && uv run process.py --db-path metrics.db

    cd benchmarks && uv run main.py --plot --db-path metrics.db --plot-save-path ./plots/ --prefix {{config_file}}

    # Move the experiment run to the training data folder
    timestamp=$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p ~/training_data/${timestamp}
    mv ./benchmarks/metrics.db ~/training_data/${timestamp}/metrics.db
    mkdir -p ~/training_data/${timestamp}/plots
    mv ./benchmarks/plots/* ~/training_data/${timestamp}/plots/

# Local version of the pipeline that doesn't require remote connections
run-local-pipeline config_file="benchmarks/configs/local.yaml" out_file="results.csv":
    mkdir -p benchmarks/plots
    go run cmd/load-generator/main.go --config {{config_file}} --out benchmarks/{{out_file}}
    ./pull-metrics-local.sh

    cd benchmarks && uv run new_import.py --csv {{out_file}} --db metrics.db
    cd benchmarks && uv run process.py --db-path metrics.db
    cd benchmarks && uv run main.py --plot --db-path metrics.db --plot-save-path ./plots/ --prefix local

    timestamp=$(date +%Y-%m-%d_%H-%M-%S)
    
allow-reuse-connections:
    # Allow reusing TIME_WAIT sockets for new connections when safe
    sudo sysctl -w net.ipv4.tcp_tw_reuse=1
watch-connections:
    watch -n 1 'ss -a | grep 10.0.0.3:50050 | wc -l'
see-connections:
    ss -a | grep 10.0.0.3:50050 | awk '{print $2}' | sort | uniq -c

############################
# Misc. Stuff
############################
# Remove all docker containers/images, and all logs
delete-logs:
    rm -rf log/*

# Print the last 100 lines of the worker log
worker-log:
    docker logs $(docker ps -a | grep worker | awk '{print $1}') --tail 100
pprof-worker:
    docker exec -it $(docker ps | grep worker | awk '{print $1}') go tool pprof http://localhost:6060/debug/pprof/goroutine
pprof-leaf:
    docker exec -it $(docker ps | grep leaf | awk '{print $1}') go tool pprof http://localhost:6060/debug/pprof/goroutine

docker-logs component:
    docker logs $(docker ps -a | grep {{component}} | awk '{print $1}') --tail 100
memory-worker:
    docker exec -it $(docker ps | grep worker | awk '{print $1}') go tool pprof http://localhost:6060/debug/pprof/heap

memory-leaf:
    docker exec -it $(docker ps | grep leaf | awk '{print $1}') go tool pprof http://localhost:6060/debug/pprof/heap
trace-worker:
    docker exec -it $(docker ps | grep worker | awk '{print $1}') go tool trace http://localhost:6060/debug/pprof/trace?seconds=60
block-worker:
    docker exec -it $(docker ps | grep worker | awk '{print $1}') go tool pprof http://localhost:6060/debug/pprof/block
clean:
    rm -rf functions/logs/*
    docker ps -a | grep hyperfaas- | awk '{print $1}' | xargs docker rm -f
    docker images | grep hyperfaas- | awk '{print $3}' | xargs docker rmi -f

#Kills the locally running integration test in case it cant shutdown gracefully
kill-worker:
    pid=$(ps aux | grep '[e]xe/worker' | awk '{print $2}') && kill -9 $pid

kill-db:
    pid=$(ps aux | grep '[e]xe/database' | awk '{print $2}') && kill -9 $pid

kill: kill-worker kill-db


plot-all:
    cd benchmarks && uv run main.py --plot --db-paths ~/model-runs/final-real/real.db ~/model-runs/final-rf/rf.db ~/model-runs/final-nn/nn.db ~/model-runs/final-linear/linear.db --plot-save-path ./plots/ --prefix final --normalize-time

plot-all2:
    cd benchmarks && uv run main.py --plot --db-paths ~/model-runs/final-real/real.db ~/model-runs/final-rf/rf.db ~/model-runs/final-linear/linear.db ~/model-runs/final-nn-2/nn-2.db --plot-save-path ./plots/ --prefix ultra --normalize-time

k6-pprof:
    go tool pprof http://localhost:6565/debug/pprof/profile?seconds=60

k6-memory:
    go tool pprof http://localhost:6565/debug/pprof/heap

linear-train db_path output:
    cd linear-training && uv run train_models.py --db {{db_path}} --output {{output}}