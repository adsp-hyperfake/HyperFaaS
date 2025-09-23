FROM golang:1.24.3-bookworm

WORKDIR /root/

# Install build dependencies for cgo and ONNX Runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        libstdc++6 \
        libgomp1 \
        python3 \
        python3-pip \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime 1.22.0 specifically to match the Go library requirement
RUN pip3 install --break-system-packages onnxruntime==1.22.0

# Find the Python site-packages directory for ONNX Runtime
RUN python3 -c "import onnxruntime; print('ONNX Runtime installed at:', onnxruntime.__file__)"

COPY . .
COPY ./cmd/worker/main.go .
COPY ./hyperFakeWorker/models /app/models

RUN go mod download

# Build the worker binary
RUN CGO_ENABLED=1 GOOS=linux go build -o main main.go


RUN mkdir -p /app/onnx-runtime

CMD ["./main", "-address=0.0.0.0:50051", "-runtime=fake-onnx", "-log-level=debug", "-log-format=text", "--auto-remove=true", "-containerized=true", "-caller-server-address=0.0.0.0:50052", "-database-type=http", "--fake-models-path=/app/models"]