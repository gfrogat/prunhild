"""
Disclaimer:

This code is based on the basic MNIST example from Pytorch examples repository
See: https://github.com/pytorch/examples/tree/master/mnist
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import prunhild

from config import parser
from utils import get_parameter_stats, print_parameter_stats


class LotteryLeNet(nn.Module):
    r"""LeNet-300-100 as used in:
    `The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks`
    https://arxiv.org/abs/1803.03635
    """

    def __init__(self):
        super(LotteryLeNet, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_dataloaders(args, kwargs):
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = (
        datasets.FashionMNIST if args.dataset == "FashionMNIST" else datasets.MNIST
    )

    train_loader = torch.utils.data.DataLoader(
        dataset(args.datafolder, train=True, download=True, transform=mnist_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    train_loader_eval = torch.utils.data.DataLoader(
        dataset(args.datafolder, train=True, download=True, transform=mnist_transform),
        batch_size=args.batch_size_eval,
        shuffle=True,
        **kwargs
    )
    test_loader_eval = torch.utils.data.DataLoader(
        dataset(args.datafolder, train=False, transform=mnist_transform),
        batch_size=args.batch_size_eval,
        shuffle=True,
        **kwargs
    )

    return train_loader, train_loader_eval, test_loader_eval


def train(
    args,
    model,
    device,
    data_loaders,
    optimizer,
    pruner,
    epoch,
    prune=False,
    prune_online=True,
):

    train_loader, train_loader_eval, test_loader_eval = data_loaders
    logs = []

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # -------------------------------------- #
        # --- Pruning Instrumentation Start --- #

        # keep already pruned weights at zero
        pruner.prune(update_state=False)

        if prune and prune_online is True:
            if batch_idx >= args.prune_delay or epoch > 1:
                if batch_idx % args.prune_interval == 0:
                    pruner.prune()

        # ---- Pruning Instrumentation End ---- #
        # -------------------------------------- #

        if batch_idx % args.print_interval == 0:
            print(
                "[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        if batch_idx % args.eval_interval == 0:
            parameter_stats = get_parameter_stats(model)
            print_parameter_stats(parameter_stats)
            _, _, ratio_zero = parameter_stats
            acc_train = evaluate(args, model, device, train_loader_eval)
            acc_test = evaluate(args, model, device, test_loader_eval)

            logs.append((epoch, batch_idx, ratio_zero, acc_train, acc_test))

    return logs


def evaluate(args, model, device, data_loader):
    fold = "Train" if data_loader.dataset.train is True else "Test"

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print(
        "[Eval] {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f})".format(
            fold, test_loss, correct, len(data_loader.dataset), accuracy
        )
    )

    return accuracy


def run_experiment(cmdline_args=None):
    args = parser.parse_args(cmdline_args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    data_loaders = setup_dataloaders(args, kwargs)

    model = LotteryLeNet().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    # --------------------------- #
    # --- Pruning Setup Start --- #

    cutoff = prunhild.cutoff.LocalRatioCutoff(args.cutoff_ratio)
    # don't prune the final bias weights
    params = list(model.parameters())[:-1]
    pruner = prunhild.pruner.CutoffPruner(params, cutoff, prune_online=True)

    # ---- Pruning Setup End ---- #
    # --------------------------- #

    logs_prune = []

    print("Pruning Start")
    torch.manual_seed(args.seed_dataloader)
    for epoch in range(1, args.epochs + 1):
        logs_prune += train(
            args, model, device, data_loaders, optimizer, pruner, epoch, prune=True
        )
    print("\n\n\n")

    # -------------------------------------- #
    # --- Pruning Weight Resetting Start --- #

    # we want to demonstrate here how to export and load the state of a pruner
    # i.e. a actual sparse model or LotteryTicket that we want to train from
    # scratch now. Make sure that the architecture and the parameters match!
    pruner_state = pruner.state_dict()

    # reset seed for initializing with the same weights
    torch.manual_seed(args.seed_retrain)
    model_retrain = LotteryLeNet().to(device)
    optimizer_retrain = optim.SGD(
        model_retrain.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
    cutoff_retrain = prunhild.cutoff.LocalRatioCutoff(args.cutoff_ratio)
    params_retrain = list(model_retrain.parameters())[:-1]
    pruner_retrain = prunhild.pruner.CutoffPruner(params_retrain, cutoff_retrain)

    # now we load the state dictionary with the prune-masks that were used last
    # for pruning the model.
    pruner_retrain.load_state_dict(pruner_state)

    # calling prune with `update_state=False` will simply apply the last prune_mask
    # stored in state
    pruner_retrain.prune(update_state=False)

    logs_retrain = []

    print("Retraining Start")
    torch.manual_seed(args.seed_dataloader_retrain)
    for epoch in range(1, args.epochs + 1):
        logs_retrain += train(
            args,
            model_retrain,
            device,
            data_loaders,
            optimizer_retrain,
            pruner_retrain,
            epoch,
            prune=False,
        )
    print("\n\n\n")

    # ---- Pruning Weight Resetting End ---- #
    # -------------------------------------- #

    return logs_prune, logs_retrain


if __name__ == "__main__":
    run_experiment()
