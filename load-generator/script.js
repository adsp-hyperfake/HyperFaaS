import grpc from 'k6/net/grpc';
import { randomSeed } from 'k6';
import { Trend } from 'k6/metrics';
import { bfsConfig, bfsSetup, bfsFunction } from './functions/bfs.js';
import { echoConfig, echoSetup, echoFunction } from './functions/echo.js';
import { thumbnailerConfig, thumbnailerSetup, thumbnailerFunction } from './functions/thumbnailer.js';

// Add our custom metrics
export const callQueuedTimestampKey = 'callqueuedtimestamp';
export const gotResponseTimestampKey = 'gotresponsetimestamp';
export const instanceIdKey = 'instanceid';
export const callQueuedTimestamp = new Trend(callQueuedTimestampKey, true);
export const gotResponseTimestamp = new Trend(gotResponseTimestampKey, true);
export const instanceIdMetric = new Trend('instanceid');

// Create global config
const config = {
  // Global Configuration
  workloadSeed: parseInt(__ENV.WORKLOAD_SEED) || Date.now(),
  totalTestDuration: __ENV.TOTAL_TEST_DURATION || "60s",
  minPreallocatedVus: parseInt(__ENV.MIN_PREALLOCATED_VUS) || 10,
  maxPreallocatedVus: parseInt(__ENV.MAX_PREALLOCATED_VUS) || 50,
  minMaxVus: parseInt(__ENV.MIN_MAX_VUS) || 20,
  maxMaxVus: parseInt(__ENV.MAX_MAX_VUS) || 100,
  rampingStartRateMin: parseInt(__ENV.RAMPING_START_RATE_MIN) || 1,
  rampingStartRateMax: parseInt(__ENV.RAMPING_START_RATE_MAX) || 5,
}

// Load executor functions for each function
// Unfortunately, ESM prohibits dynamic exports, so we have to export every function explicitly
export { bfsFunction };
export { echoFunction };
export { thumbnailerFunction };

const functionsToProcess = [
  {
    name: 'bfs',
    setup: bfsSetup,
    execFunction: bfsFunction,
    config: bfsConfig,
    exec: 'bfsFunction',
    configPrefix: 'BFS',
    imageTag: config.BFS_IMAGE_TAG
  },
  {
    name: 'echo',
    setup: echoSetup,
    execFunction: echoFunction,
    config: echoConfig,
    exec: 'echoFunction',
    configPrefix: 'ECHO',
    imageTag: config.ECHO_IMAGE_TAG
  },
  {
    name: 'thumbnailer',
    setup: thumbnailerSetup,
    execFunction: thumbnailerFunction,
    config: thumbnailerConfig,
    exec: 'thumbnailerFunction',
    configPrefix: 'THUMBNAILER',
    imageTag: config.THUMBNAILER_IMAGE_TAG
  }
];

// Initialize randomizer with workloadSeed for reproducibility
randomSeed(config.workloadSeed);

// Add function-specific configs to global config
for (const func of functionsToProcess) {
  Object.assign(config, func.config);
}

// Setup function to create the functions before the test
const client = new grpc.Client();
client.load(['./config'], 'common.proto', 'leaf.proto');

export function setup() {
  client.connect('localhost:50050', {
    plaintext: true
  });

  const setupResults = {
    client: client,
  };

  for (const func of functionsToProcess) {
    // Load setup from function file
    if (typeof func.setup === 'function') {
      const result = func.setup(client);
      setupResults[func.name] = result;
    }
  }

  return setupResults;
}

// Generate dynamic scenarios based on the functionsToProcess
const totalTestDurationSeconds = parseK6Duration(config.totalTestDuration);
let generatedScenarios = {};

for (const funcInfo of functionsToProcess) {
  const prefix = funcInfo.configPrefix;
  const minScenarios = config[`${prefix}_MIN_SCENARIOS`];
  const maxScenarios = config[`${prefix}_MAX_SCENARIOS`];

  // Skip if function is disabled or no duration
  if (maxScenarios === 0 || totalTestDurationSeconds === 0) {
    console.log(`Skipping ${funcInfo.name} scenarios (disabled or no duration)`);
    continue;
  }

  const numScenarios = getRandomInt(minScenarios, maxScenarios);
  const avgScenarioDurationSec = totalTestDurationSeconds / numScenarios;
  let currentFunctionStartTimeSec = 0;

  for (let i = 0; i < numScenarios; i++) {
    const scenarioName = `${funcInfo.name}_${i}`;
    const preAllocatedVUs = getRandomInt(config.minPreallocatedVus, config.maxPreallocatedVus);
    let maxVUs = getRandomInt(config.minMaxVus, config.maxMaxVus);
    if (maxVUs < preAllocatedVUs) maxVUs = preAllocatedVUs;

    const isConstantRate = Math.random() < config[`${prefix}_CONSTANT_SCENARIOS_RATIO`];
    const executorType = isConstantRate ? 'constant-arrival-rate' : 'ramping-arrival-rate';

    let scenarioDetails = {
      exec: funcInfo.exec,
      startTime: `${Math.floor(currentFunctionStartTimeSec)}s`,
      preAllocatedVUs: preAllocatedVUs,
      maxVUs: maxVUs,
      tags: {
        scenario_group: funcInfo.name,
        type: executorType,
        image_tag: funcInfo.imageTag
      },
      timeUnit: '1s',
    };

    if (isConstantRate) {
      scenarioDetails = {
        ...scenarioDetails,
        executor: 'constant-arrival-rate',
        duration: `${Math.ceil(avgScenarioDurationSec)}s`,
        rate: getRandomInt(
          config[`${prefix}_CONSTANT_RATE_MIN`],
          config[`${prefix}_CONSTANT_RATE_MAX`]
        )
      };
    } else {
      scenarioDetails = {
        ...scenarioDetails,
        executor: 'ramping-arrival-rate',
        startRate: getRandomInt(config.rampingStartRateMin, config.rampingStartRateMax),
        stages: [{
          target: getRandomInt(
            config[`${prefix}_BURST_TARGET_RATE_MIN`],
            config[`${prefix}_BURST_TARGET_RATE_MAX`]
          ),
          duration: `${Math.ceil(avgScenarioDurationSec)}s`
        }]
      };
    }

    generatedScenarios[scenarioName] = scenarioDetails;
    currentFunctionStartTimeSec += avgScenarioDurationSec;
  }
}

config.persistGeneration = __ENV.PERSIST_GENERATION === 'true' || false;


// Store the persistence data in a variable accessible to handleSummary
const persistenceData = {
  metadata: {
    seed: config.workloadSeed,
    totalDuration: config.totalTestDuration,
    generatedAt: new Date().toISOString(),
    configuration: config,
    /* bfsFunctionId: "bfsFunctionId",
    echoFunctionId: echoFunctionId,
    thumbnailerFunctionId: thumbnailerFunctionId */
  },
  scenarios: generatedScenarios
};

// Export K6 options
export const options = {
  scenarios: generatedScenarios,
  systemTags: ['error', 'group', 'proto', 'scenario', 'service', 'subproto', 'extra_tags', 'metadata', 'vu', 'iter']
};

// Summary function
export function handleSummary(data) {
  if (config.persistGeneration) {
    return {
      'stdout': JSON.stringify(persistenceData, null, 2)
    };
  }
  return {};
}

// Helper functions
// Function to generate a random integer between min and max
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Function to convert duration strings like "60s", "1m30s", "2h15m" to seconds
function parseK6Duration(durationStr) {
  if (typeof durationStr !== 'string') return 0;
  let totalSeconds = 0;
  const parts = durationStr.match(/(\d+h)?(\d+m)?(\d+s)?/);
  if (!parts) return 0;
  if (parts[1]) totalSeconds += parseInt(parts[1].slice(0, -1)) * 3600; // hours
  if (parts[2]) totalSeconds += parseInt(parts[2].slice(0, -1)) * 60;   // minutes
  if (parts[3]) totalSeconds += parseInt(parts[3].slice(0, -1));        // seconds
  return totalSeconds;
}

// Function to convert ISO date strings to milliseconds
export function isoToMs(isoString) {
  return new Date(isoString).getTime();
}
