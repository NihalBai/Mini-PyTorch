import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data        # actual number
        self.grad = 0.0         # gradient d(output)/d(this)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def backward(self):

        # Topological sort
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # seed gradient
        self.grad = 1.0

        # backprop
        for node in reversed(topo):
            node._backward()

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

        # Unary minus
    def __neg__(self):
        return self * -1

        # Subtraction
    def __sub__(self, other):
        return self + (-other)
    def __truediv__(self, other):
        return self * other**-1
    def sigmoid(self):
        x = self.data
        out = Value(1 / (1 + math.exp(-x)), (self,), "sigmoid")
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def log(self):
        eps = 1e-7
        x = max(self.data, eps)     # clamp only for numerical safety

        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1 / x) * out.grad

        out._backward = _backward
        return out


    def exp(self):
        import math
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

# Loss
def mse(preds, targets):
    loss = Value(0.0)
    for p, t in zip(preds, targets):
        loss = loss + (p - t) ** 2
    return loss

def bce(preds, targets):
    eps = Value(1e-7)
    loss = Value(0)
    
    for p, t in zip(preds, targets):
        p_safe = p + eps  # keep connected to graph
        loss = loss + ( -t * p_safe.log() - (Value(1.0) - t) * (Value(1.0) - p_safe).log())
    return loss



def softmax(values):
    exps = [v.exp() for v in values]
    s = sum(exps)
    return [e / s for e in exps]

# graph
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_graph(root):

    nodes, edges = trace(root)

    print("\n--- Computation Graph ---")

    for n in nodes:
        print(f"Node: data={n.data:.4f}, grad={n.grad:.4f}, op={n._op}")

    print("\n--- Graph Edges ---")
    for a, b in edges:
        print(f"{a._op or 'Input'} → {b._op}")
