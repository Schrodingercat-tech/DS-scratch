from typing import List
from collections import Counter
import numpy as np
from numpy.linalg import norm

points = List[float]


def mean(x:points)-> float:
    return sum(x)/len(x)

def _meadian_odd(x:points)->float:
    return sorted(x)[len(x)//2]

def _meadian_even(x:points)->float:
    sorted_x = sorted(x)
    hi_mid_point = len(x)//2
    return (sorted_x[hi_mid_point-1]+sorted_x[hi_mid_point])/2

def meadian(v:points)->float:
    return _meadian_even(v) if len(v)%2==0 else _meadian_odd(v)

def quantile(x:points,p:float)->float:
    p_index = int(p*len(x))
    return sorted(x)[p_index]

def mode(x:points)->points:
    counts = Counter(x)
    max_count = max(counts.values())
    return [ xi for xi,count in counts.items()
            if count == max_count]

def data_range(x:points)->float:
    return max(x)-min(x)

def mean_deviation(x:points)->points:
    x_bar = mean(x)
    return [ xi-x_bar for xi in x ]

def variance(x:points)->points:
    assert len(x)>=2,'need atleat 2 or more elements'
    n = len(x)
    deviations = mean_deviation(x)
    return norm(deviations)/(n-1)

