"""Learning rate: linear warmup, then cosine decay to min_lr. The recipes live in configs/."""
import math

def learning_rate(step, r):
    if step <= r['warmup_steps']:
        return r['peak_lr'] * step / r['warmup_steps']
    progress = (step-r['warmup_steps']) / (r['train_steps']-r['warmup_steps'])
    return r['min_lr'] + .5*(r['peak_lr']-r['min_lr'])*(1+math.cos(math.pi*progress))
