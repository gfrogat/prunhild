class _RequiredParameter(object):
    """Singleton class representing a required parameter for a Cutoff."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Cutoff(object):
    """Abstract class which provides interface for computing prune-masks.
    """

    def compute_prune_masks(self, params, prune_masks_old, grad=False):
        raise NotImplementedError("not implemented")
