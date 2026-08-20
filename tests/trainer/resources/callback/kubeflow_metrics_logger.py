"""Training Hub callback for Kubeflow TrainingHubTrainer notebooks.

Define at module level so kubeflow-sdk can serialize the class with
inspect.getsource into training pods. Pass the class (not an instance):

    callbacks=[KubeflowMetricsLogger]

Requires training_hub with unified callbacks (RHOAIENG-77626/77627).
"""

from __future__ import annotations

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext


class KubeflowMetricsLogger(TrainingHubCallback):
    """Logs key Training Hub lifecycle hooks to pod stdout (compact).

    Self-contained: no module-level references outside this class,
    so kubeflow-sdk ``inspect.getsource`` serialization works.

    Step/epoch begin/end hooks are no-ops here to keep logs readable;
    use a custom callback if you need per-step tracing.
    """

    PREFIX = "[TH-CB]"

    def _log(self, event: str, **fields: object) -> None:
        parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        msg = f"{self.PREFIX} {event}"
        if parts:
            msg = f"{msg} {parts}"
        print(msg, flush=True)

    def on_train_begin(self, context: TrainingHubContext) -> None:
        self._log("BEGIN", out=context.output_dir, main=context.is_main_process)

    def on_epoch_begin(self, context: TrainingHubContext) -> None:
        pass

    def on_step_begin(self, context: TrainingHubContext) -> None:
        pass

    def on_log(self, context: TrainingHubContext) -> None:
        self._log(
            "LOG",
            step=context.step,
            loss=context.loss,
            lr=context.learning_rate,
        )

    def on_evaluate(self, context: TrainingHubContext) -> None:
        if context.metrics:
            self._log("EVAL", step=context.step, metrics=context.metrics)

    def on_save(self, context: TrainingHubContext) -> None:
        self._log("SAVE", step=context.step, dir=context.output_dir)

    def on_step_end(self, context: TrainingHubContext) -> None:
        pass

    def on_epoch_end(self, context: TrainingHubContext) -> None:
        pass

    def on_train_end(self, context: TrainingHubContext) -> None:
        self._log("END", step=context.step)
