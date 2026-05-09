# MiniPyTorch 

A minimal educational re-implementation of core PyTorch concepts using pure Python.

This project demonstrates:
- Automatic differentiation (backpropagation)
- Computational graphs
- Neural networks (MLP)
- Activation functions
- Loss functions
- Optimizers (Adam)
- Training on the XOR classification problem

---

##  Project Structure

mini_pytorch/
│
├── minipytorch.py     Autograd engine + ops + losses
├── Neuron.py          Single neuron definition
├── Layer.py           Layer built from neurons
├── MLP.py             Multi-layer neural network
├── Adam.py            Optimizer
├── test.py            Training example (XOR)
└── README.md          Documentation


---

##  Core Components

### 1. Value (Autograd Engine)

`Value` represents a single scalar with:

- Stored data
- Gradient tracking
- Connections to parent nodes
- Automatic backpropagation using topological sorting

Supported ops:

/ **
relu tanh sigmoid
log exp

---

### 2. Neural Network (MLP)

Your MLP is fully from scratch:

- Linear layers
- Tensor-free scalar neurons
- Activation functions
- Forward inference via `model(x)`
- Gradients via `.backward()`

Example:
```python
model = MLP(2, [2, 1])

3. Loss Functions

MSE — regression loss

BCE — binary classification loss used for XOR

Example:
    loss = bce(pred, target)

4. Optimizer — Adam

Hand-written Adam update:
    optimizer = Adam(model.parameters(), lr=0.01)
    optimizer.step()

