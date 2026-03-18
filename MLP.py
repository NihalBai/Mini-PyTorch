from Layer import Layer

class MLP:
    def __init__(self, nin, layers):
        sizes = [nin] + layers
        self.layers = [
            Layer(sizes[i], sizes[i+1])
            for i in range(len(layers))
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Sigmoid ONLY on last layer
            if i == len(self.layers) - 1:
                x = [xi.sigmoid() for xi in x]

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


