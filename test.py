from minipytorch import Value, bce
from MLP import MLP
from Adam import Adam  
from minipytorch import draw_graph
# ----------------------------
# 1. XOR DATASET
# ----------------------------

data = [
    ([Value(0.0), Value(0.0)], [Value(0.0)]),
    ([Value(0.0), Value(1.0)], [Value(1.0)]),
    ([Value(1.0), Value(0.0)], [Value(1.0)]),
    ([Value(1.0), Value(1.0)], [Value(0.0)])
]

# ----------------------------
# 2. MODEL
# ----------------------------

# 2 inputs → 2 hidden neurons → 1 output neuron
model = MLP(2, [2, 1])

# ----------------------------
# 3. OPTIMIZER
# ----------------------------

optimizer = Adam(model.parameters(), lr=0.01)

# ----------------------------
# 4. TRAINING LOOP
# ----------------------------

epochs = 5000

for epoch in range(epochs):

    total_loss = Value(0)

    # ----- Forward pass -----
    for x, y in data:
        pred = model(x)
        loss = bce(pred, y)      # Binary Cross Entropy loss
        total_loss = total_loss + loss

    # ----- Backward pass -----
    total_loss.backward()
    if epoch == 0:
        draw_graph(total_loss)

    # ----- Adam parameter update -----
    optimizer.step()

    # ----- Progress logging -----
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss = {total_loss.data:.6f}")

# ----------------------------
# 5. TEST / PREDICTIONS
# ----------------------------

print("\nFinal XOR predictions:")

for x, y in data:
    pred = model(x)
    p = pred[0].data

    bool_pred = p > 0.5
    print(f"Input: {[xi.data for xi in x]} → "
          f"Prediction: {p:.4f} ({bool_pred}) "
          f"Target: {y[0].data}")
