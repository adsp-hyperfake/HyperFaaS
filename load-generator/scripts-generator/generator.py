import utils
from patterns.steady_ramp import SteadyRamp
from patterns.increasing_waves import IncreasingWaves
from patterns.constant_rate import ConstantRate
from target_function import TargetFunction


class InvalidPatternException(Exception):
    """Raised when the load pattern type does not exist."""

    def __init__(self, pattern, message="The load pattern type does not exist"):
        super().__init__(f"{message}: {pattern}")


class Generator:
    def __init__(self, config):
        self.config = config
        self.pattern_generators = {}
        self.target_functions = []
        self.setups = {}
        self.vus = []

    def initialize(self):
        self.create_pattern_generators()
        self.create_target_functions()

    def create_pattern_generators(self):
        for pattern_config in self.config["load_patterns"]:
            match pattern_config["type"]:
                case "steady_ramp":
                    self.pattern_generators[pattern_config["name"]] = SteadyRamp()
                case "increasing_waves":
                    self.pattern_generators[pattern_config["name"]] = IncreasingWaves()
                case "constant_rate":
                    self.pattern_generators[pattern_config["name"]] = ConstantRate()
                case _:
                    raise InvalidPatternException(pattern_config["type"])

    def create_target_functions(self):
        for function_config in self.config["functions"]:
            self.target_functions.append(TargetFunction(
                function_config["name"], function_config["endpoint"], function_config["parameters"]
            ))

    def generate(self):
        for setup_config in self.config["setups"]:
            if isinstance(setup_config["load_patterns"], list):
                patterns = setup_config["load_patterns"]
            else:
                patterns = list(self.pattern_generators.keys())
            if isinstance(setup_config["functions"], list):
                functions = setup_config["functions"]
            else:
                functions = self.target_functions
            self.vus.append((setup_config["preallocated_vus"], setup_config["max_vus"]))
            self.setups[setup_config["name"]] = []
            for _ in range(setup_config["count"]):
                stages = utils.get_random_int(**setup_config["stages"])
                self.generate_setup(
                    setup_config["name"],
                    setup_config["duration"],
                    setup_config["rps"],
                    stages,
                    patterns,
                    functions,
                )

    def generate_setup(
        self, name: str, duration: dict, rps: dict, stages: int, patterns: list, functions: list
    ):
        generated_scenarios = [[] for _ in range(len(functions))]
        for _ in range(stages):
            stage_duration = utils.get_random_int(**duration)
            for ind in range(len(functions)):
                func_rps = utils.get_random_int(**rps)
                pattern = utils.get_random_list_entry(patterns)
                generated_scenarios[ind].append(self.pattern_generators[pattern].generate_stage(stage_duration, func_rps))
        self.setups[name].append(generated_scenarios)

    def get_setups(self):
        return self.setups

    def get_functions(self):
        return self.target_function
