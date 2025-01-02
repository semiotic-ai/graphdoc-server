# system packages
import logging

# internal packages
from graphdoc import GraphNetworkArbitrum

# external packages
import pytest
from subgrounds import Subgraph as SubgroundsSubgraph

logging.basicConfig(level=logging.INFO) 

class TestSubgraph:

    def test_subgraph_init(self, sg: GraphNetworkArbitrum):
        assert isinstance(sg.subgraph, SubgroundsSubgraph)