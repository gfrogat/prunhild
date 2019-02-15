import argparse

defaults = {
    "no_cuda": False,
    "seed": 2,
    "seed_dataloader": 2,
    "seed_retrain": 2,
    "seed_dataloader_retrain": 2,
    "datafolder": "~/data/torch",
    "dataset": "FashionMNIST",
    "epochs": 1,
    "batch_size": 32,
    "batch_size_eval": 512,
    "learning_rate": 1e-2,
    "momentum": 0.8,
    "cutoff_ratio": 0.15,
    "prune_interval": 200,
    "prune_delay": 200,
    "print_interval": 200,
    "eval_interval": 200,
}

parser = argparse.ArgumentParser(description="`prunhild` MNIST example")

# General
parser.add_argument("--no-cuda", action="store_true", help="disables CUDA training")
parser.add_argument(
    "--seed",
    type=int,
    metavar="S",
    help="random seed (default: {})".format(defaults["seed"]),
)
parser.add_argument(
    "--seed-dataloader",
    type=int,
    metavar="S",
    help="random seed dataloader (default: {})".format(defaults["seed_dataloader"]),
)
parser.add_argument(
    "--seed-retrain",
    type=int,
    metavar="S",
    dest="seed_retrain",
    help="random seed for retraining (default: {})".format(defaults["seed_retrain"]),
)
parser.add_argument(
    "--seed-dataloader-retrain",
    type=int,
    metavar="S",
    dest="seed_dataloader_retrain",
    help="random seed dataloader for retraining (default: {})".format(
        defaults["seed_dataloader_retrain"]
    ),
)

# Dataset
parser.add_argument(
    "--datafolder",
    type=str,
    metavar="PATH_TO_FOLDER",
    help="path to datafolder (default: {})".format(defaults["datafolder"]),
)
parser.add_argument(
    "--dataset",
    type=str,
    metavar="DATASET",
    help="MNIST variant to use (default: {})".format(defaults["dataset"]),
)

# Hyperparameters
hyperparameters = parser.add_argument_group("hyperparameters")
hyperparameters.add_argument(
    "--epochs",
    type=int,
    metavar="N",
    help="number of epochs to train (default: {})".format(defaults["epochs"]),
)
hyperparameters.add_argument(
    "--batch-size",
    type=int,
    metavar="N",
    help="batch size for training (default: {})".format(defaults["batch_size"]),
)
hyperparameters.add_argument(
    "--batch-size-eval",
    type=int,
    metavar="N",
    help="batch size for evaluating (default: {})".format(defaults["batch_size_eval"]),
)
hyperparameters.add_argument(
    "--lr",
    type=float,
    metavar="LR",
    dest="learning_rate",
    help="learning rate (default: {})".format(defaults["learning_rate"]),
)
hyperparameters.add_argument(
    "--momentum",
    type=float,
    metavar="M",
    help="SGD momentum (default: {})".format(defaults["momentum"]),
)

# Cutoff
cutoff = parser.add_argument_group("cutoff")
cutoff.add_argument(
    "--cutoff-ratio",
    type=float,
    metavar="RATIO",
    help="cutoff ratio (default: {})".format(defaults["cutoff_ratio"]),
)

pruning = parser.add_argument_group("pruning")
pruning.add_argument(
    "--prune-interval",
    type=int,
    metavar="N",
    help="wait N batches between pruning steps (default: {})".format(
        defaults["prune_interval"]
    ),
)
pruning.add_argument(
    "--prune-delay",
    type=int,
    metavar="N",
    help="wait N batches before starting with pruning (default: {})".format(
        defaults["prune_delay"]
    ),
)

# Logging
logging = parser.add_argument_group("logging")
logging.add_argument(
    "--print-interval",
    type=int,
    metavar="N",
    help="during training print metrics every N batches (default: {})".format(
        defaults["print_interval"]
    ),
)
logging.add_argument(
    "--eval_interval",
    type=int,
    metavar="N",
    help="log performance of model every N batches (default: {})".format(
        defaults["eval_interval"]
    ),
)

parser.set_defaults(**defaults)
