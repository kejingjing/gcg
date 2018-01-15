"""
How to use:
(1) Create a template file
(2) For all parameters you wish to iterate over, replace with a unique "V_<name>"
(3) Fill in the below TODO
"""

import os
import re
import itertools

# TODO What is the template file?
BASE_EXP_NAME = 'rw_rccar/var'
BASE_EXP_NUM = 000

# TODO What are all the keys being replaced?
option_keys = ['V_height', 'V_width', 'V_color', 'V_H', 'V_K', 'V_device', 'V_frac']

# TODO What are all the combinations you want to do?
imsize = [{'V_height': height, 'V_width': width} for (height, width) in [(36, 64), (90, 160)]]
imsize = [
    {'V_height': 36, 'V_width': 64,  'V_device': 1, 'V_frac': 0.4},
    {'V_height': 90, 'V_width': 160, 'V_device': 0, 'V_frac': 0.4},
]
color = [{'V_color': c} for c in (1, 3)]
horizon = [{'V_H': H} for H in (12, 16)]
action = [{'V_K': K} for K in (2048, 4096)]

product_options = (imsize, color, horizon, action)

# TODO How many seeds do you want to do for each combination?
num_seeds = 1

######################
### Load base file ###
######################

def exp_name(num, seed):
    return '{0}{1:03d}/seed_{2}'.format(BASE_EXP_NAME, num, seed)

def exp_yaml_name(num, seed=None):
    s = '{0}{1:03d}'.format(BASE_EXP_NAME, num)
    if seed is not None:
        s += '_s{0}'.format(seed)
    s += '.yaml'
    return s

assert(os.path.exists(exp_yaml_name(BASE_EXP_NUM)))
with open(exp_yaml_name(BASE_EXP_NUM), 'r') as f:
    base_text = f.read()

for i, product in enumerate(itertools.product(*product_options)):
    curr_exp_num = BASE_EXP_NUM + 1 + i
    for curr_exp_seed in range(num_seeds):
        curr_exp_name = exp_name(curr_exp_num, curr_exp_seed)
        curr_exp_yaml_name = exp_yaml_name(curr_exp_num, curr_exp_seed)

        ### combine options into one dictionary
        curr_options = {}
        for option in product:
            curr_options.update(option)

        ### make sure no key was left out
        assert(tuple(sorted(option_keys)) == tuple(sorted(curr_options.keys())))

        print(curr_exp_name + ' | ' + ', '.join(['{0}: {1}'.format(key, curr_options[key]) for key in option_keys]))

        ### replace instances in the text
        curr_text = base_text
        for key, value in curr_options.items():
            curr_text = curr_text.replace(key, str(value))

        assert('V_' not in curr_text)

        ### replace exp_name and seed
        curr_text = re.sub('exp_name:.*\n', 'exp_name: {0}\n'.format(curr_exp_name), curr_text)
        curr_text = re.sub('seed:.*\n', 'seed: {0}\n'.format(curr_exp_seed), curr_text)

        ### write to file
        with open(curr_exp_yaml_name, 'w') as f:
            f.write(curr_text)
        
