import sys
sys.path.insert(0,r'./')
import random
import torch
import argparse
import numpy as np


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


class ParseKwargsOptim(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        kwargs_dict = getattr(namespace, self.dest)

        # Check if "optim_name" key is provided
        optim_name_value = next((value for value in values if value.startswith("optim_name=")), None)
        if optim_name_value:
            kwargs_dict["optim_name"] = optim_name_value.split("=")[1]
        else:
            # Find the first argument that is not in the key-value format
            non_key_value_args = [value for value in values if "=" not in value]
            if non_key_value_args:
                kwargs_dict["optim_name"] = non_key_value_args[0]
            else:
                raise "Please provide the optimizer name!"

        for value in values:
            if "=" in value:
                key, value = value.split('=')

                # Skip the "optim_name" key
                if key == "optim_name":
                    continue

                # Try converting value to int
                try:
                    value = int(value)
                except ValueError:
                    # Try converting value to float
                    try:
                        value = float(value)
                    except ValueError:
                        # Try converting value to tuple of floats or ints
                        try:
                            value = tuple(map(float, value.split(',')))
                        except ValueError:
                            pass  # Value cannot be converted, keep it as a string

                kwargs_dict[key] = value


def set_seed(value):
    print("\n Random Seed: ", value)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)


def clear_cuda_cache():
    print("\n Clearing cuda cache...")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargsOptim)
    args = parser.parse_args()
    print(args.kwargs)