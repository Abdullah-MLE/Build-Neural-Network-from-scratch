import sys, math
from nn_project.nn.activations import sigmoid
for raw in sys.stdin:
    line = raw.rstrip("\n").strip()
    if not line: continue

    x_str, w_str, b_str = line.split(";")
    x = list(map(float, x_str.split(",")))
    w = list(map(float, w_str.split(",")))
    b = float(b_str)

    # TODO: compute z = dot(w, x) + b, then print sigmoid(z) = 1 / (1 + exp(-z))
    # rounded to 4 decimals.
    z = sum(a * b for a, b in zip(w, x)) + b
    print(f"{sigmoid(z):.4f}")
