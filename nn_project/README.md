# Neural Network From Scratch

A simple learning project built lesson by lesson with Python.

## Where each lesson goes

| Course topic | File |
|---|---|
| Neurons and dense layers | `nn/layers.py` |
| Sigmoid, ReLU, and softmax | `nn/activations.py` |
| MSE and cross-entropy | `nn/losses.py` |
| Manual forward and backward passes | `nn/model.py` |
| Gradient descent, momentum, and Adam | `nn/optimizers.py` |
| Scalar autograd | `nn/autograd.py` |
| L2 and dropout | `nn/regularization.py` |
| Xavier and Kaiming initialization | `nn/initialization.py` |
| Batch and layer normalization | `nn/normalization.py` |
| MNIST loading and batches | `nn/data.py` |
| Training and evaluation | `nn/training.py` |
| Conv2D and max pooling | `nn/vision.py` |
| Embeddings and positional encoding | `nn/embeddings.py` |
| RNN cell | `nn/recurrent.py` |
| Attention and transformer block | `nn/attention.py` |

Start with `nn/layers.py`, `nn/activations.py`, `nn/losses.py`, and `nn/model.py`. Add the logic from each lesson to the matching file, then connect everything in `nn/training.py`.

`main.py` in the repository root remains reserved for course submissions and is intentionally untouched.

## Running the learning project

From the repository root, run the project with Python:

```sh
python -c "from nn_project.nn.layers import Neuron"
```

When you create your own practice file inside `nn_project`, import the neuron like this:

```python
from nn.layers import Neuron
```

Run that file from inside `nn_project` with `python your_file.py`.
