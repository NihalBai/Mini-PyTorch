import math

class Adam:
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        # moments
        self.m = {p: 0.0 for p in params}
        self.v = {p: 0.0 for p in params}

        self.t = 0

    def step(self):
        self.t += 1

        for p in self.params:
            g = p.grad

            # update biased moments
            self.m[p] = self.b1 * self.m[p] + (1 - self.b1) * g
            self.v[p] = self.b2 * self.v[p] + (1 - self.b2) * (g * g)

            # bias corrections
            m_hat = self.m[p] / (1 - self.b1 ** self.t)
            v_hat = self.v[p] / (1 - self.b2 ** self.t)

            # parameter update
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            # clear gradient
            p.grad = 0
