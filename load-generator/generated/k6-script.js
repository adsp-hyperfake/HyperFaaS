import grpc from 'k6/net/grpc';
import { check } from 'k6';

// K6 GRPC client
const client = new grpc.Client();

// Load all proto files
const protoFiles = ["definitions.proto","common.proto"];
client.load(['../config'], ...protoFiles);

// K6 Options
export const options = {
  "MyFunction_t1": {
    "executor": "constant-arrival-rate",
    "startTime": "0s",
    "exec": "MyFunction_exec",
    "duration": "2s",
    "rate": 20,
    "preAllocatedVUs": 40,
    "timeUnit": "1s",
    "maxVUs": 40
  },
  "MyFunction_t2": {
    "executor": "constant-arrival-rate",
    "startTime": "2s",
    "exec": "MyFunction_exec",
    "duration": "3s",
    "rate": 30,
    "preAllocatedVUs": 60,
    "timeUnit": "1s",
    "maxVUs": 60
  },
  "MyFunction_t3": {
    "executor": "constant-arrival-rate",
    "startTime": "5s",
    "exec": "MyFunction_exec",
    "duration": "NaNs",
    "rate": null,
    "preAllocatedVUs": null,
    "timeUnit": "1s",
    "maxVUs": null
  }
};

// Service call functions
__EXEC_FUNCTIONS__
