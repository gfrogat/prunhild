# prunhild
*prunhild* is a small library for neural network pruning based on [PyTorch](https://pytorch.org).

I wrote this library for better structuring my code when replicating the experiments from [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) for a university course.

The library currently only implements simple magnitude-based pruning as used in the previously mentioned paper.
Througout the library magnitude-pruning is referred to as cutoff-pruning.

The library offers a simple class CutoffPruner which offers a similar interface as the [Optimizer](https://pytorch.org/docs/stable/optim.html) class from PyTorch.
The Pruner class takes different strategies (Cutoffs) for computing the binary masks (prune-masks) used for pruning the weights.


## Getting Started

### Installation
You can install the library directly from GitHub:
```bash
# Install from GitHub
pip install git+https://github.com/gfrogat/prunhild
```

## Examples
The folder [examples](./examples) contains an example using MNIST on how to use the library.
If you encounter problems or have any questions don't hesitate to open an issue.

- prunhild example on MNIST - [code](./examples/mnist)
