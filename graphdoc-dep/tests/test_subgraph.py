# system packages
import os
import yaml
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphNetworkArbitrum

# external packages
import pytest
import pickle
from pandas import DataFrame
from subgrounds import Subgraph as SubgroundsSubgraph

logging.basicConfig(level=logging.INFO)

# set global variables
SUBGRAPH_ID = "DZz4kDTdmzWLWsV373w2bSmoar3umKKH9y82SUKr5qmp"
CACHE_DIR = Path(__file__).parent / "assets/cache/"


class TestSubgraph:

    def test_subgraph_init(self, sg: GraphNetworkArbitrum):
        assert isinstance(sg.subgraph, SubgroundsSubgraph)

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_query_subgraph_ipfs(self, sg: GraphNetworkArbitrum, request):
        """
        file_name = "test_query_subgraph_ipfs.pkl"
        cache_pick_file = {
            "subgraph_ipfs_response": subgraph_ipfs_response,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_query_subgraph_ipfs.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        subgraph_ipfs_response = None

        if dry_fire:
            if os.path.exists(test_cache_file_path):
                with open(test_cache_file_path, "rb") as f:
                    cache = pickle.load(f)
                    subgraph_ipfs_response = cache.get("subgraph_ipfs_response")
            else:
                raise FileNotFoundError(
                    f"Cache file not found at {test_cache_file_path} for --dry-fire mode"
                )

        if fire:
            subgraph_ipfs_response = sg.query_subgraph_ipfs(SUBGRAPH_ID)
            cache = {"subgraph_ipfs_response": subgraph_ipfs_response}
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)

        #################### testing ####################
        assert subgraph_ipfs_response is not None, "subgraph_ipfs_response is not set"
        assert isinstance(subgraph_ipfs_response, DataFrame)

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_get_subgraph_ipfs_hash(self, sg: GraphNetworkArbitrum, request):
        """
        file_name = "test_get_subgraph_ipfs_hash.pkl"
        cache_pick_file = {
            "subgraph_ipfs_hash": subgraph_ipfs_hash,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_get_subgraph_ipfs_hash.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        subgraph_ipfs_hash = None

        if dry_fire:
            if os.path.exists(test_cache_file_path):
                with open(test_cache_file_path, "rb") as f:
                    cache = pickle.load(f)
                    subgraph_ipfs_hash = cache.get("subgraph_ipfs_hash")
            else:
                raise FileNotFoundError(
                    f"Cache file not found at {test_cache_file_path} for --dry-fire mode"
                )

        if fire:
            subgraph_ipfs_hash = sg.get_subgraph_ipfs_hash(SUBGRAPH_ID)
            cache = {"subgraph_ipfs_hash": subgraph_ipfs_hash}
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)

        #################### testing ####################
        assert subgraph_ipfs_hash is not None, "subgraph_ipfs_hash is not set"
        assert isinstance(subgraph_ipfs_hash, str)

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_get_subgraph_ipfs_manifest(self, sg: GraphNetworkArbitrum, request):
        """
        file_name = "test_get_subgraph_ipfs_manifest.pkl"
        cache_pick_file = {
            "subgraph_ipfs_manifest": subgraph_ipfs_manifest,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_get_subgraph_ipfs_manifest.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        subgraph_ipfs_manifest = None

        if dry_fire:
            if os.path.exists(test_cache_file_path):
                with open(test_cache_file_path, "rb") as f:
                    cache = pickle.load(f)
                    subgraph_ipfs_manifest = cache.get("subgraph_ipfs_manifest")
            else:
                raise FileNotFoundError(
                    f"Cache file not found at {test_cache_file_path} for --dry-fire mode"
                )

        if fire:
            subgraph_ipfs_manifest = sg.get_subgraph_ipfs_manifest(SUBGRAPH_ID)
            cache = {"subgraph_ipfs_manifest": subgraph_ipfs_manifest}
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)

        #################### testing ####################
        assert subgraph_ipfs_manifest is not None, "subgraph_ipfs_manifest is not set"
        assert isinstance(subgraph_ipfs_manifest, dict)

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_find_matching_values(self, sg: GraphNetworkArbitrum, request):
        """
        file_name = "test_find_matching_values.pkl"
        cache_pick_file = {
            "matching_values": matching_values,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_find_matching_values.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        matching_values = None

        if dry_fire:
            if os.path.exists(test_cache_file_path):
                with open(test_cache_file_path, "rb") as f:
                    cache = pickle.load(f)
                    matching_values = cache.get("matching_values")
            else:
                raise FileNotFoundError(
                    f"Cache file not found at {test_cache_file_path} for --dry-fire mode"
                )

        if fire:
            subgraph_ipfs_manifest = sg.get_subgraph_ipfs_manifest(SUBGRAPH_ID)
            matching_values = sg.find_matching_values(subgraph_ipfs_manifest)
            cache = {"matching_values": matching_values}
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)

        #################### testing ####################
        assert matching_values is not None, "matching_values is not set"
        assert isinstance(matching_values, list)
        assert len(matching_values) > 0

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_get_abis_hashes_from_manifest(self, sg: GraphNetworkArbitrum, request):
        """
        file_name = "test_get_abis_hashes_from_manifest.pkl"
        cache_pick_file = {
            "abis_hashes": abis_hashes,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_get_abis_hashes_from_manifest.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        abis_hashes = None

        if dry_fire:
            if os.path.exists(test_cache_file_path):
                with open(test_cache_file_path, "rb") as f:
                    cache = pickle.load(f)
                    abis_hashes = cache.get("abis_hashes")
            else:
                raise FileNotFoundError(
                    f"Cache file not found at {test_cache_file_path} for --dry-fire mode"
                )

        if fire:
            abis_hashes = sg.get_abis_hashes_from_manifest(SUBGRAPH_ID)
            cache = {"abis_hashes": abis_hashes}
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)

        #################### testing ####################
        assert abis_hashes is not None, "abis_hashes is not set"
        assert isinstance(abis_hashes, list)
        assert len(abis_hashes) > 0

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_get_abis_from_manifest(self, sg: GraphNetworkArbitrum, request):
        """
        file_name = "test_get_abis_from_manifest.pkl"
        cache_pick_file = {
            "abis": abis,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_get_abis_from_manifest.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        abis = None

        if dry_fire:
            if os.path.exists(test_cache_file_path):
                with open(test_cache_file_path, "rb") as f:
                    cache = pickle.load(f)
                    abis = cache.get("abis")
            else:
                raise FileNotFoundError(
                    f"Cache file not found at {test_cache_file_path} for --dry-fire mode"
                )

        if fire:
            abis = sg.get_abis_from_manifest(SUBGRAPH_ID)
            cache = {"abis": abis}
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)

        #################### testing ####################
        assert abis is not None, "abis is not set"
        assert isinstance(abis, dict)
        assert len(abis) > 0
