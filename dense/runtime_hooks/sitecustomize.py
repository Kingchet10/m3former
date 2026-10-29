"""Runtime hook for scheduler-launched jobs (outside the experiment code, so run identities are unchanged).

TensorBoard event steps are protobuf int64. The research tracker also logs against cumulative nominal training
FLOPs, which pass 2**63 late in an XL run and crash the writer. Clamp event steps to the int64 maximum; the exact
counters stay in research-steps.jsonl and in checkpoints."""
import importlib.abc
import importlib.util
import sys

_TARGET = 'torch.utils.tensorboard.writer'
_INT64_MAX = 2 ** 63 - 1


class _ClampTensorboardSteps(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def patched(module):
            exec_module(module)
            add_event = module.FileWriter.add_event

            def clamped(self, event, step=None, walltime=None):
                if step is not None and step > _INT64_MAX:
                    step = _INT64_MAX
                return add_event(self, event, step, walltime)

            module.FileWriter.add_event = clamped
            init = module.SummaryWriter.__init__

            def init_clamped(self, *args, **kwargs):
                if kwargs.get('purge_step') is not None and kwargs['purge_step'] > _INT64_MAX:
                    kwargs['purge_step'] = _INT64_MAX
                return init(self, *args, **kwargs)

            module.SummaryWriter.__init__ = init_clamped

        spec.loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _ClampTensorboardSteps())
