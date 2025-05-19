class TargetFunction:
    def __init__(self, name, endpoint, parameters):
        self.name = name
        self.endpoint = endpoint
        self.parameters = parameters

    def get_param_names(self):
        return [param["name"] for param in self.parameters]

    def get_parameters(self):
        return self.parameters