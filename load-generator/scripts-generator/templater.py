import textwrap

from generator import Generator

class Templater:

    JS_TEMPLATE = textwrap.dedent("""\
    import grpc from 'k6/net/grpc';
    import {{ check }} from 'k6';
    import {{ randomItem }} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';


    // K6 GRPC client
    const client = new grpc.Client();
    client.load(['../config'], '__PROTO_FILE__');

    // K6 Options
    export const options = {{
        scenarios: {{
            {scenarios_for_template}
        }}
    }};

    // Parameters
    const params = {{
        {params_for_template}
    }}

    // Service call functions
    {functions_for_template}""")

    SCENARIO_TEMPLATE=textwrap.dedent("""\
        scenario_t{scenario_index}: {{
                    executor: 'ramping-arrival-rate',
                    startTime: '0s',
                    exec: '{scenario_exec}',
                    preAllocatedVUs: '{scenario_prealloc_vus}',
                    maxVUs: '{scenario_max_vus}',
                    stages: [
                        {scenario_stages}
                    ],
                }}""")

    FUNCTION_TEMPLATE=textwrap.dedent("""\
    export function {function_template_name}() {{
        client.connect('localhost:50052', {{ plaintext: true }});

        // payload
        {function_template_payload}

        const response = client.invoke('{function_template_endpoint}', payload);

        check(response, {{
            'status is OK': (r) => r && r.status === grpc.StatusOK
        }});
    }}""")

    PAYLOAD_TEMPLATE=textwrap.dedent("""\
    {params}

        const payload = {{
            {payload}
        }}""")

    def __init__(self, generator: Generator, path: str):
        self.setups = generator.setups
        self.functions = generator.target_functions
        self.vus = generator.vus
        self.path = path

    def write_to_file(self):
        for setup_ind, name in enumerate(self.setups.keys()):
            target_functions = self.assemble_functions()
            prealloc_vus = self.vus[setup_ind][0]
            max_vus = self.vus[setup_ind][1]
            for ind, setup in enumerate(self.setups[name]):
                scenarios = self.assemble_scenarios(setup, prealloc_vus, max_vus)
                params = self.assemble_params()
                js_data = self.assemble_js_file(scenarios, target_functions, params)
                with open(self.path+f"{ind}.js", "w") as file:
                 file.write(js_data)


    def assemble_scenarios(self, setup, prealloc_vus, max_vus):
        stages = []
        function_names = [ function.name for function in self.functions ]
        for stage in setup:
            stages.append(",\n                ".join(stage))
        formatted_scenarios = [ self.SCENARIO_TEMPLATE.format(scenario_index=ind, scenario_exec=function_names[ind], scenario_stages=scenario, scenario_prealloc_vus=prealloc_vus, scenario_max_vus=max_vus) for ind, scenario in enumerate(stages)]
        return ",\n        ".join(formatted_scenarios)

    def assemble_functions(self):
        function_names = [ f.name for f in self.functions ]
        function_endpoints = [ f.endpoint for f in self.functions ]
        param_names = [ f.get_param_names() for f in self.functions]
        payloads = self.assemble_payload(function_names, param_names)
        assembled_funcs = []
        for name, payload, endpoint in zip(function_names, payloads, function_endpoints):
            assembled_funcs.append(self.FUNCTION_TEMPLATE.format(function_template_name=name, function_template_payload=payload, function_template_endpoint=endpoint))
        return "\n\n".join(assembled_funcs)

    def assemble_params(self):
        params = [ f.get_parameters() for f in self.functions]
        param_strings = []
        for ind, f in enumerate(self.functions):
            formatted_strings = [f"{param['name']}: {param['values']}" for param in params[ind]]
            param_strings.append(f"{f.name}: {{ " + ", ".join(formatted_strings) + " }")
        return ",\n    ".join(param_strings)

    def assemble_js_file(self, scenarios, target_functions, params):
        return self.JS_TEMPLATE.format(scenarios_for_template=scenarios, functions_for_template=target_functions, params_for_template=params)

    def assemble_payload(self, function_names, param_names):
        param_strings = [[] for _ in range(len(function_names))]
        payload_strings = [[] for _ in range(len(function_names))]
        for ind, name in enumerate(function_names):
            for param in param_names[ind]:
                param_strings[ind].append(f"const {param} = randomItem(params.{name}.{param});")
                payload_strings[ind].append(f"{param}: {param}")
        payloads = []
        for ind in range(len(function_names)):
            payloads.append(self.PAYLOAD_TEMPLATE.format(params="\n    ".join(param_strings[ind]), payload=",\n        ".join(payload_strings[ind])))
        return payloads