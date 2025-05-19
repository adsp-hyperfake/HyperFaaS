# k6 scripts generator

This is a work in progress – testing, documentation and other features will soon be added.

TODO: Proto file.

The generator is based on a YAML file ([see][0]) in which the general settings are outlined.

## Settings

The YAML file is based on three dictionaries: `functions`, `load_patterns` and `setups`.

### Functions

Here we define the functions that k6 will trigger.


| Parameter  | Description                                                                 | Required | Default |
|------------|-----------------------------------------------------------------------------|:--------:|:-------:|
| name       | A unique name of the function.                                              |   ✔️     |    ❌  |
| endpoint   | The endpoint used to call the function.                                     |   ✔️     |    ❌  |
| parameters | A dictionary of parameters, each parameter has a name and a list of values. |   ✔️     |    ❌  |


### load_patterns


| Parameter | Description                        | Required | Default|
|-----------|------------------------------------|:--------:|:-------|
| name      | A unique name of the load pattern. |    ✔️    |    ❌  |
| type      | The pattern type.                  |    ✔️    |    ❌  |

This is a work in progress feature. For now, only the `steady_ramp` pattern is implemented and it takes no further
parameters.

## setups

Here we can define different setups that define how many scripts are generated and the respective settings.

| Parameter        | Description                                                                                     | Required | Default |
|------------------|-------------------------------------------------------------------------------------------------|:--------:|:-------:|
| name             | A unique name for the setup.                                                                    |   ✔️     |   ❌    |
| count            | The number of scripts to generate.                                                              |   ✔️     |   ❌    |
| duration         | How long (seconds) each stage should last. Dict of min, max and mean (triangular distribution). |   ✔️     |   ❌    |
| rps              | The amount of requests per second (triangular distribution as above).                           |   ✔️     |   ❌    |
| stages           | How many stages are to be generated (triangular distribution as above).                         |   ✔️     |   ❌    |
| load_patterns    | The list of load_patterns to use. Allows for the shorthand "all".                               |   ✔️     |   ❌    |
| functions        | The list of functions to use. Allows for the shorthand "all".                                   |   ✔️     |   ❌    |
| max_vus          | The maximum number of VUs.                                                                      |   ✔️     |   ❌    |
| preallocated_vus | The preallocated number of VUs.                                                                 |   ✔️     |   ❌    |

## General generation process

For a given setup, a random choice for the number of stages will be made. Next, for each stage, a random duration will
be chosen.

For each function, a separate scenario will be created and for each stage a random `rps` value will be chosen for each
function.

The parameters will be randomly selected from the list of provided values for each request.

## How to run the script

First, run `pip install -r ./load-generator/scripts-gnerator/requirements.txt` in the root of the project. Next, execute
`python ./load-generator/scripts-generator/generate_setups.py`. The `example.yml` and the output directory are currently
very basically hardcoded. If you run into issues, just change the lines. This will be changed soon.

[0]: "./example.yml"