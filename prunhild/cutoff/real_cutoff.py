from .cutoff import Cutoff, required
from .utils import binary_mask_weight, binary_mask_grad


# TODO add checks for generators
class RealCutoff(Cutoff):
    """Strategy for computing the cutoff-values used for generating the
    prune-masks.

    Args:
        cutoff_value (float): weights smaller the specified cutoff-value will
            be pruned.

    """

    def __init__(self, cutoff_value=required):
        if cutoff_value is required:
            raise ValueError(
                "Didn't specify a value for required parameter cutoff-value"
            )
        if cutoff_value is not required and cutoff_value < 0.0:
            raise ValueError(
                "Cutoff value has to be positive. Value: {}".format(cutoff_value)
            )

        self.cutoff_value = cutoff_value

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        format_string += "cutoff_value={}".format(self.cutoff_value)
        format_string += ")"
        return format_string

    def compute_prune_masks(self, params, prune_masks_old, grad=False):
        params = list(params)

        binary_mask = binary_mask_weight if grad is False else binary_mask_grad

        for param, prune_mask in zip(params, prune_masks_old):
            if param.requires_grad is False:
                yield None
            else:
                assert param.data.size() == prune_mask.size()
                prune_mask = binary_mask(param, prune_mask, self.cutoff_value)
                yield prune_mask
