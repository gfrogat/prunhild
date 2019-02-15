import torch
from torch.optim.optimizer import Optimizer, required


# TODO: keep reset_params only with online pruning?
class CutoffPruner(Optimizer):
    """Implements magnitude-based pruning using different strategies for computing
    the prune-masks.

    The CutoffPruner object is initialized with a Cutoff, a strategy for computing
    a binary-mask used for pruning the model.

    In a pruning step the object computes the prune-masks using the initially
    defined strategy (e.g. LocalRatioCutoff: for every layer the smallest <ratio>
    percent of unpruned weights will get masked).

    The prune-masks are then used for pruning the corresponding weights
    (they are set to zero).

    Args:
        params (iterable): iterable of the parameters to prune or dicts defining
            parameter groups.
        cutoff (Cutoff): a Cutoff projects which implements computation of the
            binary masks (prune-masks) used for pruning.
        grad_cutoff (bool, optional): optionally prune the weights based on the
            magnitude of the gradients (default: False).
        prune_online (bool, optional): prune online (i.e. prune multiple times
            during training). When online pruning the every pruning step the
            prune-masks will mask ratio percent of the unpruned weights and the
            weights which were already pruned in a previous step (default: False).
        reset_params (bool, optional): when online reset the weights to their
            initial values at t_{0}. This is the strategy used in the
            `Lottery Ticket Hypothesis Paper`__ (default: True).

    Example usage::

        local_ratiocutoff = LocalRatioCutoff(ratio=0.05)
        pruner = CutoffPruner(model.parameters(), cutoff=local_ratiocutoff)
        pruner.prune()

    __ https://arxiv.org/abs/1803.03635

    .. note::
        When pruning CutoffPruner will prune all parameters which require a gradient.
        This might lead to problems if the model contains batch normalization layers
        which pruning will generally break. When in doubt make sure to pass the
        weights as separate parameter groups.
    """

    def __init__(
        self,
        params,
        cutoff=required,
        grad_cutoff=False,
        prune_online=False,
        reset_params=True,
    ):
        defaults = dict(
            cutoff=cutoff,
            grad_cutoff=grad_cutoff,
            prune_online=prune_online,
            reset_params=reset_params,
        )

        super(CutoffPruner, self).__init__(params, defaults)

        for group in self.param_groups:
            params = group["params"]
            prune_online = group["prune_online"]
            reset_params = group["reset_params"]

            for param in params:
                param_state = self.state[param]
                if "prune_mask" not in param_state:
                    param_state["prune_mask"] = torch.ones_like(param.data)

                if prune_online and reset_params is True:
                    if "param_init" not in param_state:
                        param_state["param_init"] = param.data.clone().detach()

    def __setstate__(self, state):
        super(CutoffPruner, self).__setstate__(state)

    def _prune_by_mask(self, params, prune_masks):
        for param, prune_mask in zip(params, prune_masks):
            if param.requires_grad is False:
                assert prune_mask is None
                continue

            assert param.data.size() == prune_mask.size()
            param.data.mul_(prune_mask)

    def _reset_params(self, params):
        for param in params:
            param_state = self.state[param]
            param.data.copy_(param_state["param_init"])

    def prune_groups(self, update_masks=False):
        prune_groups = list()

        for group in self.param_groups:
            params = group["params"]
            cutoff = group["cutoff"]
            grad_cutoff = group["grad_cutoff"]
            prune_masks_old = list()

            for param in params:
                param_state = self.state[param]
                prune_masks_old.append(param_state["prune_mask"])

            if update_masks is True:
                prune_masks = cutoff.compute_prune_masks(
                    params, prune_masks_old, grad=grad_cutoff
                )
                prune_masks = list(prune_masks)
            else:
                prune_masks = prune_masks_old

            prune_group = {
                "cutoff": cutoff,
                "grad_cutoff": grad_cutoff,
                "prune_masks": prune_masks,
            }
            prune_groups.append(prune_group)

        return prune_groups

    def prune(self, update_state=True):
        prune_groups = self.prune_groups(update_state)
        self.prune_by_prune_groups(prune_groups, update_state)

    def prune_by_prune_groups(self, prune_groups, update_state=True):
        groups = self.param_groups
        if len(groups) != len(prune_groups):
            raise ValueError(
                "prune_masks group has differnt number of parameter groups"
            )

        param_lens = (len(g["params"]) for g in groups)
        prune_mask_lens = (len(g["prune_masks"]) for g in prune_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, prune_mask_lens)):
            raise ValueError(
                "prune_masks group contains prune_mask "
                "that doesn't match the size of pruner's group"
            )

        for group, prune_group in zip(groups, prune_groups):
            params = group["params"]
            prune_online = group["prune_online"]
            reset_params = group["reset_params"]
            prune_masks = prune_group["prune_masks"]

            if update_state is True:
                for param, prune_mask in zip(params, prune_masks):
                    param_state = self.state[param]
                    param_state["prune_mask"] = prune_mask

                if prune_online and reset_params is True:
                    self._reset_params(params)

            self._prune_by_mask(params, prune_masks)
