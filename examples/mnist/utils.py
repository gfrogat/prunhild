def get_parameter_stats(model):
    n_zero = 0.0
    n_total = 0.0
    for param in model.parameters():
        # assume values smaller than 1e-7 (for 32bit) to be zero
        n_zero += param.data.abs().le(1e-7).sum().item()
        n_total += param.data.numel()

    ratio_zero = n_zero / n_total
    return n_zero, n_total, ratio_zero


def print_parameter_stats(parameter_stats):
    n_zero, n_total, ratio_zero = parameter_stats
    print(
        "[Model] parameters zero: ({} / {} | {:.2f})".format(
            n_zero, n_total, ratio_zero
        )
    )
