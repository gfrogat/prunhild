import numpy as np
from .cutoff import Cutoff, required
from .utils import binary_mask_weight, binary_mask_grad


# TODO add checks for generators
class RatioCutoff(Cutoff):
    """Abstract class which implements the logic for computation of prune-masks
    based on the ratio of number of weights. The prune-masks are generated by
    masking all weight values that are smaller than a specific cutoff.

    Args:
        ratio (float): ratio of weights that should be pruned

    The computation of the cutoff-values is implemented by its derived classes.
    """

    def __init__(self, ratio=required):
        if ratio is required:
            raise ValueError("Didn't specify a value for required parameter ratio")
        if ratio is not required and (ratio < 0.0 or ratio > 1.0):
            raise ValueError(
                "Ratio value has to between 0.0 and 1.0. Value: {}".format(ratio)
            )

        self.ratio = ratio

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        format_string += "ratio={}".format(self.ratio)
        format_string += ")"
        return format_string

    def compute_prune_masks(self, params, prune_masks_old, grad=False):
        """Compute prune-masks for all parameters tensors

        Args:
            params (iterable): iterable of the parameters to prune
            prune_masks_old (iterable): iterable of prune-masks

        This will output a generator which will yield an updated prune-mask for
        every parameter tensor.
        """

        params = list(params)

        binary_mask = binary_mask_weight if grad is False else binary_mask_grad
        cutoff_values = self.compute_cutoff_values(params, prune_masks_old)

        for param, prune_mask, cutoff_value in zip(
            params, prune_masks_old, cutoff_values
        ):
            if param.requires_grad is False:
                assert cutoff_value is None
                yield None
            else:
                assert param.data.size() == prune_mask.size()
                prune_mask = binary_mask(param, prune_mask, cutoff_value)
                yield prune_mask

    def compute_cutoff_values(self, params, prune_masks_old):
        raise NotImplementedError("Not implemented")


class LocalRatioCutoff(RatioCutoff):
    """Strategy for generating the prune-mask, by masking all values that are
    smaller than a cutoff-value.

    Args:
        ratio (float): ratio of weights that should be pruned

    The cutoff-values are computed for every layer, where the cutoff-value is the
    ratio * 100 percentile of the parameter values.
    """

    def compute_cutoff_values(self, params, prune_masks_old):
        """Compute cutoff-value per layer

        Args:
            params (iterable): iterable of the parameters to prune
            prune_masks_old (iterable): iterable of prune-masks

        This will output a list with the cutoff-values for every parameter tensor
        of a parames iterable.
        """

        cutoff_values = list()

        for param, prune_mask in zip(params, prune_masks_old):
            if param.requires_grad is False:
                cutoff_values.append(None)
                continue
            weights = param.data[prune_mask > 0.5].abs().flatten().cpu().numpy()
            if len(weights) != 0:
                cutoff_values.append(np.percentile(weights, self.ratio * 100.0))
            else:
                cutoff_values.append(param.data.abs().max().item())

        return cutoff_values


class GlobalRatioCutoff(RatioCutoff):
    """Strategy for generating the prune-masks by masking all values that are
    smaller than a cutoff-value.

    Args:
        ratio (float): ratio of weights that should be pruned

    The cutoff-values are computed over all layers, where the cutoff-value is the
    ratio * 100 percentile of the parameter values in all layers.
    """

    def compute_cutoff_values(self, params, prune_masks_old):
        """Compute cutoff-value across all layers

        Args:
            params (iterable): iterable of the parameters to prune
            prune_masks_old (iterable): iterable of prune-masks

        This will output a list with the cutoff-values for every parameter tensor
        of a parames iterable.
        """

        n_weights = 0

        # compute number of items for allocating numpy array
        for param, prune_mask in zip(params, prune_masks_old):
            if param.requires_grad is False:
                continue
            n_weights += (prune_mask > 0.5).sum()

        # allocate array
        weights = np.empty((n_weights, 1))
        index = 0

        # copy weights into numpy array
        for param, prune_mask in zip(params, prune_masks_old):
            if param.requires_grad is False:
                continue
            n_param = (prune_mask > 0.5).sum()
            if n_param > 0:
                weights[index : (index + n_param)] = (
                    param.data[prune_mask > 0.5].view(n_param, -1).cpu().numpy()
                )
                index += n_param

        # compute cutoff value
        global_cutoff_value = np.percentile(np.abs(weights), self.ratio * 100)
        del (weights)

        # create list of global cutoff values
        cutoff_values = list()
        for param in params:
            if param.requires_grad is False:
                cutoff_values.append(None)
                continue
            cutoff_values.append(global_cutoff_value)

        return cutoff_values
