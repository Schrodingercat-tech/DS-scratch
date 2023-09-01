import numpy as np
from typing import List

vector = List[float]

class linalg:
    @staticmethod
    def add(a:vector,b:vector)->vector:
        assert len(a)== len(b),"two vectors must have same dimension"
        return np.array(a)+np.array(b)
    
    @staticmethod
    def subtract(a:vector,b:vector)->vector:
        assert len(a)==len(b),"two vectors must have same dimension"
        return np.array(a)-np.array(b)

    @staticmethod
    def vector_sum(vectors : List[vector])->vector:
        assert vectors,' no vectors provided'
        dimension = len(vectors[0])
        assert all(len(v)==dimension for v in vectors),' All vectors must have the same dimension'
        new_vector = np.zeros(dimension)
        for vec in vectors:
            new_vector+=np.array(vec)
        return new_vector

    @staticmethod
    def scalar_multiply(c:float,a:vector)->vector:
        return c*np.array(a)

    @staticmethod
    def vector_mean(vectors:List[vector])->vector:
        dimension = len(vectors)
        return linalg.scalar_multiply(1/dimension,linalg.vector_sum(vectors))
   
    @staticmethod
    def dot (v:vector,w:vector)->float:
        assert len(v)==len(w),'both vectors must be of same dimension'
        return np.dot(v,w)
  
    @staticmethod
    def sum_of_squares(a:vector)->float:
        return np.sum(np.array(a)**2)
  
    @staticmethod
    def magnitude(a:vector)->float:
        return np.sqrt(linalg.sum_of_squares(a))
 
    @staticmethod
    def squared_distance(v:vector,w:vector)->float:
        return linalg.sum_of_squares(linalg.subtract(v,w))
  
    @staticmethod
    def distance(v:vector,w:vector)->float:
        return linalg.magnitude(linalg.subtract(v,w))

