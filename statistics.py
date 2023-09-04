from typing import List
from collections import Counter
from numpy.linalg import norm
import numpy as np
from linalg import linalg

points = List[float]

class stats:

    @staticmethod
    def mean(x:points)-> float:
        return sum(x)/len(x)
    @staticmethod
    def _meadian_odd(x:points)->float:
        return sorted(x)[len(x)//2]
    @staticmethod
    def _meadian_even(x:points)->float:
        sorted_x = sorted(x)
        hi_mid_point = len(x)//2
        return (sorted_x[hi_mid_point-1]+sorted_x[hi_mid_point])/2
    @staticmethod
    def meadian(v:points)->float:
        return stats._meadian_even(v) if len(v)%2==0 else stats._meadian_odd(v)
    @staticmethod
    def quantile(x:points,p:float)->float:
        p_index = int(p*len(x))
        return sorted(x)[p_index]
    @staticmethod
    def mode(x:points)->points:
        counts = Counter(x)
        max_count = max(counts.values())
        return [ xi for xi,count in counts.items()
                if count == max_count]
    @staticmethod
    def data_range(x:points)->float:
        return max(x)-min(x)
    @staticmethod
    def de_mean(x:List)->List[float]:
        x_bar  =stats.mean(x)
        return [i-x_bar for i in x]
    @staticmethod
    def mean_deviation(x:points)->points:
        x_bar = stats.mean(x)
        return [ xi-x_bar for xi in x ]
    @staticmethod
    def variance(x:points)->points:
        assert len(x)>=2,' need at least 2 or more elements
        n = len(x)
        deviations = stats.mean_deviation(x)
        return norm(deviations)/(n-1)
    @staticmethod
    def standard_deviation(x:List[float])->float:
        return np.sqrt(stats.variance(x))
    @staticmethod
    def interquartile_range(x:List[float])->float:
        return stats.quantile(x,0.75)-stats.quantile(x,0.25)
    @staticmethod
    def covariance(x:List[float],y:List[float])->float:
        assert len(x)==len(y),"x and y must be of same dimension"
        return linalg.dot(stats.de_mean(x),stats.de_mean(y))/(len(x)-1)
    @staticmethod
    def correlation(x:List[float],y:List[float])->float:
        stdevx = stats.standard_deviation(x)
        stdevy = stats.standard_deviation(y)
        if stdevx>0 and stdevy>0: return stats.covariance(x,y)/(stdevx*stdevy)
        else: return 0
   
