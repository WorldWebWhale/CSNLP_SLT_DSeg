# schedulers.py  (or top of notebook)
import numpy as np

class LambdaWarmUpCosineScheduler:
    """
    Multiplicative LR schedule:
        * linear warm-up from lr_start → lr_max for `warm_up_steps`
        * cosine decay from lr_max → lr_min for the rest of the run
    Use with torch.optim.lr_scheduler.LambdaLR(base_lr = lr_max).
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start,
                 max_decay_steps, verbosity_interval=0):
        self.warm_up_steps     = warm_up_steps
        self.lr_min            = lr_min
        self.lr_max            = lr_max
        self.lr_start          = lr_start
        self.max_decay_steps   = max_decay_steps
        self.last_lr           = 0.
        self.verbose_every     = verbosity_interval

    def __call__(self, step: int):
        # optional debug print
        if self.verbose_every and step % self.verbose_every == 0:
            print(f"[sched] step {step} – mult {self.last_lr:.4e}")

        if step < self.warm_up_steps:                      # warm-up
            lr_mult = (self.lr_max - self.lr_start) / self.warm_up_steps * step + self.lr_start
        else:                                              # cosine decay
            t = (step - self.warm_up_steps) / max(1, self.max_decay_steps - self.warm_up_steps)
            t = min(t, 1.0)
            lr_mult = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * t))

        self.last_lr = lr_mult
        return lr_mult
