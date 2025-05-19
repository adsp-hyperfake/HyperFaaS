class SteadyRamp:
    def __init__(self):
        pass

    def generate_stage(self, duration, rps):
        return f"{{ duration: '{duration}s', target: {rps} }}"
