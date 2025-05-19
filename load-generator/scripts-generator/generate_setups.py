import os

import yaml

from generator import Generator
from templater import Templater

def read_yaml_file(path):
    with open(path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            os._exit(1)


if __name__ == "__main__":
    config = read_yaml_file(path="./load-generator/scripts-generator/example.yml")
    generator = Generator(config)
    generator.initialize()
    generator.generate()
    templater = Templater(generator, "./load-generator/scripts-generator/results/")
    templater.write_to_file()
