import math
import sys

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        val = float(line)
        result = sigmoid(val)
        print(f"{result:.4f}")


if __name__ == "__main__":
    main()