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
                { duration: '75s', target: 460 },
                { duration: '52s', target: 456 },
                { duration: '32s', target: 634 },
                { duration: '34s', target: 719 },
                { duration: '10s', target: 768 },
                { duration: '25s', target: 696 },
                { duration: '73s', target: 693 },
                { duration: '32s', target: 771 },
                { duration: '54s', target: 651 },
                { duration: '23s', target: 595 },
                { duration: '6s', target: 224 },
                { duration: '55s', target: 596 },
                { duration: '48s', target: 369 },
                { duration: '27s', target: 669 },
                { duration: '77s', target: 440 },
                { duration: '51s', target: 329 },
                { duration: '12s', target: 260 },
                { duration: '47s', target: 300 },
                { duration: '38s', target: 409 },
                { duration: '5s', target: 600 },
                { duration: '45s', target: 755 },
                { duration: '78s', target: 644 },
                { duration: '45s', target: 756 },
                { duration: '42s', target: 709 },
                { duration: '15s', target: 161 },
                { duration: '51s', target: 741 },
                { duration: '9s', target: 631 },
                { duration: '47s', target: 942 },
                { duration: '35s', target: 578 },
                { duration: '14s', target: 561 },
                { duration: '47s', target: 654 },
                { duration: '42s', target: 353 },
                { duration: '43s', target: 739 },
                { duration: '13s', target: 843 },
                { duration: '40s', target: 270 },
                { duration: '42s', target: 794 },
                { duration: '95s', target: 369 },
                { duration: '84s', target: 649 },
                { duration: '22s', target: 719 },
                { duration: '29s', target: 496 },
                { duration: '14s', target: 786 },
                { duration: '78s', target: 751 }
            ],
        },
        scenario_t1: {
            executor: 'ramping-arrival-rate',
            startTime: '0s',
            exec: 'func_2',
            preAllocatedVUs: '10',
            maxVUs: '75',
            stages: [
                { duration: '75s', target: 941 },
                { duration: '52s', target: 654 },
                { duration: '32s', target: 615 },
                { duration: '34s', target: 467 },
                { duration: '10s', target: 586 },
                { duration: '25s', target: 699 },
                { duration: '73s', target: 749 },
                { duration: '32s', target: 284 },
                { duration: '54s', target: 188 },
                { duration: '23s', target: 946 },
                { duration: '6s', target: 317 },
                { duration: '55s', target: 818 },
                { duration: '48s', target: 771 },
                { duration: '27s', target: 773 },
                { duration: '77s', target: 362 },
                { duration: '51s', target: 492 },
                { duration: '12s', target: 221 },
                { duration: '47s', target: 208 },
                { duration: '38s', target: 821 },
                { duration: '5s', target: 458 },
                { duration: '45s', target: 469 },
                { duration: '78s', target: 307 },
                { duration: '45s', target: 291 },
                { duration: '42s', target: 636 },
                { duration: '15s', target: 713 },
                { duration: '51s', target: 985 },
                { duration: '9s', target: 857 },
                { duration: '47s', target: 851 },
                { duration: '35s', target: 470 },
                { duration: '14s', target: 706 },
                { duration: '47s', target: 604 },
                { duration: '42s', target: 846 },
                { duration: '43s', target: 454 },
                { duration: '13s', target: 699 },
                { duration: '40s', target: 348 },
                { duration: '42s', target: 319 },
                { duration: '95s', target: 894 },
                { duration: '84s', target: 863 },
                { duration: '22s', target: 490 },
                { duration: '29s', target: 746 },
                { duration: '14s', target: 824 },
                { duration: '78s', target: 655 }
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