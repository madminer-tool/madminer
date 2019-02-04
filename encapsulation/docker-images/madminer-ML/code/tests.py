import numpy as np
import yaml
import sys

inputs_file = sys.argv[1]

with open(inputs_file) as f:
    inputs = yaml.safe_load(f)

print(inputs)
print(inputs['prior'])

print(inputs['prior']['parameter_0']['prior_shape'])

