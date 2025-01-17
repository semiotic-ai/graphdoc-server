# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality, DocQualityEval

# external packages
import pytest
from dspy import Example
from dspy import LM, Predict

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

class TestDocQualityEval: 

    def test_validate_category(self): 
        ex = Example()
        ex.category = "test"
        
        dqe = DocQualityEval()
        assert dqe.validate_category(ex, ex)

    def test_validate_rating(self): 
        ex = Example()
        ex.rating = "test"
        
        dqe = DocQualityEval()
        assert dqe.validate_rating(ex, ex)

    