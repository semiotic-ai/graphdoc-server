# system packages

# internal packages
from .evaluate import DocQuality

# external packages
import dspy


class GraphDoc:
    def __init__(
        self,
        model: str,
        api_key: str,
        cache: bool = True,
    ) -> None:

        # initialize base dspy config
        self.lm = dspy.LM(model=model, api_key=api_key, cache=cache)
        dspy.configure(lm=self.lm)

        # initialize modules
        self.doc_eval = dspy.Predict(DocQuality)
