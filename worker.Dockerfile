ARG GO_VERSION=1.24.3
ARG ALPINE_VERSION=latest

FROM golang:${GO_VERSION}-alpine

WORKDIR /root/

# Install build dependencies for cgo
RUN apk add --no-cache gcc musl-dev

COPY . .
COPY ./cmd/worker/main.go .

RUN go mod download

#Copy main function
RUN CGO_ENABLED=1 GOOS=linux go build -o main main.go

CMD ["./main", "-address=0.0.0.0:50051", "-runtime=docker", "-log-level=debug", "-log-format=text", "--auto-remove=true", "-containerized=true", "-caller-server-address=0.0.0.0:50052", "-database-type=http"]
