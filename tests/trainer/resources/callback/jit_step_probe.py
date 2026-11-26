"""Training Hub callback used by the JIT checkpoint e2e notebook.

Prints one line per optimizer step so the notebook can tell when to send
SIGTERM and which steps a resumed job trained. Defined at module level and
passed as a class so kubeflow-sdk can serialize it with inspect.getsource.
"""

from __future__ import annotations

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext


class JITStepProbe(TrainingHubCallback):
    """Logs ``[JIT-E2E] step=<n> epoch=<e>`` after every optimizer step."""

    def on_step_end(self, context: TrainingHubContext) -> None:
        print(f"[JIT-E2E] step={context.step} epoch={context.epoch}", flush=True)
