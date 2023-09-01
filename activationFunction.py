import numpy as np

class activate:
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x: float) -> float:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def relu(x: float) -> float:
        return np.maximum(0, x)

    @staticmethod
    def prelu(x: float, alpha: float) -> float:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.5) -> float:
        return np.maximum(alpha * x, x)

    @staticmethod
    def elu(x: float, alpha: float = 1.0) -> float:
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    @staticmethod
    def step_function(x: float) -> float:
        return 0.0 if x < 0 else 1.0

    @staticmethod
    def swish(x: float, beta: float) -> float:
        return x * activate.sigmoid(beta * x)

    @staticmethod
    def identity(x: float) -> float:
        return x

    @staticmethod
    def gaussian(x:float, mu=0, sigma=1):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def bent_identity(x:float):
        return ((np.sqrt(x ** 2 + 1) - 1) / 2) + x

    @staticmethod
    def softplus(x:float):
        return np.log(1 + np.exp(x))

    @staticmethod
    def arctan(x:float):
        return np.arctan(x)

    @staticmethod
    def isrlu(x:float, alpha=1.0):
        return np.where(x > 0, x, x / np.sqrt(1 + alpha * x ** 2))

    @staticmethod
    def softsign(x:float):
        return x / (1 + np.abs(x))

    @staticmethod
    def custom_activation(x:float):
        return np.sin(x) / x

    @staticmethod
    def sqnl(x:float):
        def piecewise(x):
            return np.where(x > 2.0, 1.0, x - x ** 2 / 4)

        return np.where(x >= 0, piecewise(x), -piecewise(-x))

    @staticmethod
    def mish(x:float):
        return x * np.tanh(np.log(1 + np.exp(x)))

    @staticmethod
    def bent_exp(x:float):
        return ((np.sqrt(x ** 2 + 1) - 1) / 2) + x + np.exp(-x) - 1

    @staticmethod
    def selu(x:float, alpha=1.67326, scale=1.0507):
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def soft_exponential(x:float, alpha=1.0):
        return np.where(x > 0, alpha * (np.exp(x) - 1), -alpha * np.exp(-x) + 1)


