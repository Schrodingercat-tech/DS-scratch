import numpy as np
from collections import defaultdict,Counter
from typing import List

vector = List[float]

def add(a:vector,b:vector)->vector:
    assert len(a)== len(b),"two vectors must have same dimension"
    return np.array(a)+np.array(b)

def subtract(a:vector,b:vector)->vector:
    assert len(a)==len(b),"two vectors must have same dimension"
    return np.array(a)-np.array(b)

def vector_sum(vectors : List[vector])->vector:
    assert vectors,'no vectors provided'
    dimension = len(vectors[0])
    assert all(len(v)==dimension for v in vectors),'All vectors must have same dimension'
    new_vector = np.zeros(dimension)
    for vec in vectors:
        new_vector+=np.array(vec)
    return new_vector

def scalar_multiply(c:float,a:vector)->vector:
    return c*np.array(a)

def vector_mean(vectors:List[vector])->vector:
    dimension = len(vectors)
    return scalar_multiply(1/dimension,vector_sum(vectors))

def dot (v:vector,w:vector)->float:
    assert len(v)==len(w),'both vectors must be of same dimension'
    return np.dot(v,w)

def sum_of_squares(a:vector)->float:
    return np.sum(np.array(a)**2)

def magnitude(a:vector)->float:
    return np.sqrt(sum_of_squares(a))

def squared_distance(v:vector,w:vector)->float:
    return sum_of_squares(subtract(v,w))

def distance(v:vector,w:vector)->float:
    return magnitude(subtract(v,w))

