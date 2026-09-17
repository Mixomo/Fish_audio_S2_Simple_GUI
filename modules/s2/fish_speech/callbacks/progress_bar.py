from __future__ import annotations

from collections import defaultdict

from lightning.pytorch.callbacks import Callback
from tqdm.auto import tqdm


class GradAccumProgressBar(Callback):
    """Optimizer-step progress + TensorBoard bridge.

    Micro-batches are intentionally hidden. All training scalars are averaged
    across the accumulation window and emitted exactly once at the resulting
    ``trainer.global_step``. Validation scalars are emitted once at validation
    end using that same optimizer-step axis.
    """

    TRAIN_KEYS = (
        "train/loss",
        "train/base_loss",
        "train/semantic_loss",
        "train/top_5_accuracy",
        "train/grad_norm",
    )
    VAL_KEYS = (
        "val/loss",
        "val/base_loss",
        "val/semantic_loss",
        "val/top_5_accuracy",
    )
    DISPLAY_KEYS = ("train/loss", "train/top_5_accuracy", "val/loss", "val/top_5_accuracy")

    def __init__(self):
        super().__init__()
        self.bar = None
        self._last_step = 0
        self._metric_sum = defaultdict(float)
        self._metric_count = defaultdict(int)
        self._last_val_step = -1

    @staticmethod
    def _scalar(value):
        if value is None:
            return None
        try:
            return float(value.detach().cpu())
        except Exception:
            try:
                return float(value)
            except Exception:
                return None

    @staticmethod
    def _writer(trainer):
        experiment = getattr(getattr(trainer, "logger", None), "experiment", None)
        return experiment if experiment is not None and hasattr(experiment, "add_scalar") else None

    def _collect_microbatch_metrics(self, trainer):
        for key in self.TRAIN_KEYS:
            value = self._scalar(trainer.callback_metrics.get(key))
            if value is not None:
                self._metric_sum[key] += value
                self._metric_count[key] += 1

    def _flush_optimizer_step(self, trainer, step):
        averaged = {}
        for key in self.TRAIN_KEYS:
            count = self._metric_count.get(key, 0)
            if count:
                averaged[key] = self._metric_sum[key] / count

        writer = self._writer(trainer) if trainer.is_global_zero else None
        if writer is not None:
            for key, value in averaged.items():
                writer.add_scalar(key, value, int(step))
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                opt_name = optimizer.__class__.__name__
                for pg_idx, group in enumerate(optimizer.param_groups, start=1):
                    if "lr" in group:
                        suffix = f"/opt{opt_idx + 1}" if len(trainer.optimizers) > 1 else ""
                        writer.add_scalar(
                            f"lr-{opt_name}/pg{pg_idx}{suffix}",
                            float(group["lr"]),
                            int(step),
                        )
            if hasattr(writer, "flush"):
                writer.flush()

        self._metric_sum.clear()
        self._metric_count.clear()
        return averaged

    def _postfix(self, trainer, train_values=None):
        values = dict(train_values or {})
        for key in self.VAL_KEYS:
            value = self._scalar(trainer.callback_metrics.get(key))
            if value is not None:
                values[key] = value
        return {key: f"{values[key]:.4g}" for key in self.DISPLAY_KEYS if key in values}

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        total = max(1, int(trainer.max_steps))
        initial = max(0, min(total, int(trainer.global_step)))
        self._last_step = initial
        self.bar = tqdm(
            total=total, initial=initial, desc="Optimizer steps", unit="step",
            dynamic_ncols=True, leave=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect_microbatch_metrics(trainer)
        step = int(trainer.global_step)
        if step <= self._last_step:
            return

        train_values = self._flush_optimizer_step(trainer, step)
        if trainer.global_rank == 0 and self.bar is not None:
            bounded = min(step, int(trainer.max_steps))
            delta = bounded - self._last_step
            if delta > 0:
                self.bar.update(delta)
            self._last_step = bounded
            postfix = self._postfix(trainer, train_values)
            if postfix:
                self.bar.set_postfix(postfix, refresh=True)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        step = int(trainer.global_step)
        if step == self._last_val_step:
            return
        values = {}
        for key in self.VAL_KEYS:
            value = self._scalar(trainer.callback_metrics.get(key))
            if value is not None:
                values[key] = value
        writer = self._writer(trainer) if trainer.is_global_zero else None
        if writer is not None and values:
            for key, value in values.items():
                writer.add_scalar(key, value, step)
            if hasattr(writer, "flush"):
                writer.flush()
        self._last_val_step = step
        if trainer.global_rank == 0 and self.bar is not None:
            postfix = self._postfix(trainer)
            if postfix:
                self.bar.set_postfix(postfix, refresh=True)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0 and self.bar is not None:
            step = min(int(trainer.global_step), int(trainer.max_steps))
            if step > self._last_step:
                self.bar.update(step - self._last_step)
            self.bar.close()
            self.bar = None

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0 and self.bar is not None:
            self.bar.close()
            self.bar = None
