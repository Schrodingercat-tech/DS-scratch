from linalg import vector,add,scalar_multiply
from typing import Callable,TypeVar,List,Iterator
import random


# derivates

def difference_quotient(
        f:Callable[[float],float],
        x:float,
        h:float=0.0001)->float:
    return (f(x+h)-f(x))/h

def partial_difference_quotient(
        f:Callable[[vector],float],
        v:vector,
        i:int,
        h:float = 0.0001)->float:
    w = [vj+(h if j==i else 0)
         for j,vj in enumerate(v)]
    return (f(w)-f(v))/h

def estimate_gradient(
        f:Callable[[vector],float],
        v:vector,
        h:float = 0.0001):
    return [partial_difference_quotient(f,v,i,h)
            for i in range(len(v))]

# using Gradient step

def gradient_step(
        v:vector,
        gradient:vector,
        step_size:float):
    assert len(v)==len(gradient),'dim must be same'
    step = scalar_multiply(step_size,gradient)
    return add(v,step)

def sum_of_squares_gradient(v:vector):
    return [2*vi for vi in v]

# use gradient decent to fit the model

def linear_gradient(x:float,y:float,theta:vector)->vector:
    slope,intercept = theta
    predicted = slope*x+intercept
    error = (predicted-y)
    grad = [2*error*x,2*error]
    return grad

# minibatch 

T = TypeVar('T') # for writing 'generic' functions

def mminibatches(
        dataset:List[T],
        batch_size:int,
        shuffle:bool=True)->Iterator[List[T]]:
    batch_starts = [start  for start in range(0,len(dataset),batch_size)]
    if shuffle : random.shuffle(batch_starts)
    for start in batch_starts:
        end = start+batch_size
        yield dataset[start:end]
