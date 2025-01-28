# system packages
import logging

# internal packages
from graphdoc import GraphDoc

# external packages

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestGraphdoc:
    def test_graphdoc(self, gd: GraphDoc):
        assert isinstance(gd, GraphDoc)

    # def test__initialize_trainer(self, gd: GraphDoc):
