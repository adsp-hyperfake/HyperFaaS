import grpc from 'k6/net/grpc';
import { check } from 'k6';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';


// K6 GRPC client
const client = new grpc.Client();
client.load(['../config'], '__PROTO_FILE__');

// K6 Options
export const options = {
    scenarios: { 
        scenario_t0: {
            executor: 'ramping-arrival-rate',
            startTime: '0s',
            exec: 'func_1',
            preAllocatedVUs: '10',
            maxVUs: '75',
            stages: [
                { duration: '71s', target: 351 },
                { duration: '33s', target: 444 },
                { duration: '36s', target: 784 },
                { duration: '74s', target: 210 },
                { duration: '44s', target: 799 },
                { duration: '43s', target: 678 },
                { duration: '4s', target: 620 },
                { duration: '28s', target: 300 },
                { duration: '25s', target: 307 }
            ],
        },
        scenario_t1: {
            executor: 'ramping-arrival-rate',
            startTime: '0s',
            exec: 'func_2',
            preAllocatedVUs: '10',
            maxVUs: '75',
            stages: [
                { duration: '71s', target: 930 },
                { duration: '33s', target: 754 },
                { duration: '36s', target: 810 },
                { duration: '74s', target: 688 },
                { duration: '44s', target: 562 },
                { duration: '43s', target: 750 },
                { duration: '4s', target: 551 },
                { duration: '28s', target: 797 },
                { duration: '25s', target: 737 }
            ],
        }
    }
};

// Parameters
const params = { 
    func_1: { param1: [1.5, 2, 2.5, 3, 3.5], param2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] },
    func_2: { param1: [1, 2, 3], param2: [10] }
}

// Service call functions
export function func_1() {
    client.connect('localhost:50052', { plaintext: true });

    // payload
    const param1 = randomItem(params.func_1.param1);
    const param2 = randomItem(params.func_1.param2);

    const payload = {
        param1: param1,
        param2: param2                              
    }

    const response = client.invoke('api/func1', payload);

    check(response, {
        'status is OK': (r) => r && r.status === grpc.StatusOK
    });
}

export function func_2() {
    client.connect('localhost:50052', { plaintext: true });

    // payload
    const param1 = randomItem(params.func_2.param1);
    const param2 = randomItem(params.func_2.param2);

    const payload = {
        param1: param1,
        param2: param2                              
    }

    const response = client.invoke('api/func1', payload);

    check(response, {
        'status is OK': (r) => r && r.status === grpc.StatusOK
    });
}