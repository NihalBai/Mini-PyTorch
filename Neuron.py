import random
from minipytorch import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        out = self.b
        for wi, xi in zip(self.w, x):
            out = out + wi * xi
        return out.relu()

    def parameters(self):
        return self.w + [self.b]
