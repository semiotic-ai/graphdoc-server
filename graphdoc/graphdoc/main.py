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

    # def update_graphdoc_dataset(): 
        # load the dataset from the repo files 
        # optionally, let another location be specified to pull data from 
        # drop duplicates 
        # require a version, dataset card, and commit message 
        # push to the repo 
        # return the version number and the commit SHA
