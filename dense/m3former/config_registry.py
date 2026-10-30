"""TorchTitan config registry: `--module m3former --config wsd`."""


def wsd():
    from .recipe import recipe

    return recipe()
