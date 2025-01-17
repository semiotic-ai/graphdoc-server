# system packages

# internal packages 

# external packages 
import dspy

class GraphDoc: 
    def __init__(self, 
                 model: str,
                 api_key: str,
                 ) -> None:
        
        self.lm = dspy.LM(model=model, api_key=api_key)
        dspy.configure(lm=self.lm)