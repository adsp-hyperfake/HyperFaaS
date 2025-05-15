import grpc from 'k6/net/grpc';
import { check } from 'k6';

// K6 GRPC client
const client = new grpc.Client();
client.load(['../config'], '__PROTO_FILE__');

// K6 Options
export const options = __SCENARIOS__;

// Service call functions
__EXEC_FUNCTIONS__
