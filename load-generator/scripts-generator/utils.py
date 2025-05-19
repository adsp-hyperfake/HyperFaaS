import random
from scipy.stats import triang
import numpy as np


def get_random_int(min, max, **kwargs):
    if not kwargs:
        random_int = random.randint(min, max)
    elif "mean" in kwargs:
        mean = kwargs["mean"]
        mode = 3 * mean - min - max
        mode = np.clip(mode, min, max)
        random_int = triang.rvs(
            c=(mode - min) / (max - min), loc=min, scale=(max - min)
        )
    return round(random_int)


def get_random_list_entry(list):
    return random.choice(list)
