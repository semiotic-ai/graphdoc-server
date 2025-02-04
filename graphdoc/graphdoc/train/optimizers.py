# system packages
import logging
import inspect
from typing import Any, Dict

# internal packages

# external packages
import dspy

# logging
log = logging.getLogger(__name__)


def _optimizer_kwargs_filter(init_signature, optimizer_kwargs: Dict[str, Any]):
    filtered_kwargs = {
        k: v for k, v in optimizer_kwargs.items() if k in init_signature.parameters
    }
    return filtered_kwargs


def optimizer_class(optimizer_type: str, optimizer_kwargs: Dict[str, Any]):
    if optimizer_type == "miprov2":
        # metric: Callable (this is a function that takes a prediction and an example)
        # auto: str ("light", "medium", "heavy")
        # ...
        return dspy.MIPROv2(
            **_optimizer_kwargs_filter(
                init_signature=inspect.signature(dspy.MIPROv2.__init__),
                optimizer_kwargs=optimizer_kwargs,
            )
        )
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")


def optimizer_compile(optimizer_type: str, optimizer_kwargs: Dict[str, Any]):
    optimizer = optimizer_class(optimizer_type, optimizer_kwargs)
    # miprov2
    # student: dspy.ChainOfThought, dspy.Predict, etc.
    # trainset: List[dspy.Example]
    # max_labeled_demos: int
    # max_bootstrapped_demos: int
    return optimizer.compile(
        **_optimizer_kwargs_filter(
            init_signature=inspect.signature(optimizer.compile),
            optimizer_kwargs=optimizer_kwargs,
        )
    )
