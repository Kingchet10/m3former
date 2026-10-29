# FLA kernel-only namespace for this experiment.
# Original FLA copyright: Songlin Yang, Yu Zhang, Zhiyuan Li, 2023-2026, MIT.
# Do not import optional Transformers model wrappers in a Megatron application.
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
__version__ = '0.6.0'
