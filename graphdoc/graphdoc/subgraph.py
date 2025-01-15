# system packages
import os
import yaml
import json
import logging
import requests
from typing import Optional, List, Dict, Any

# local packages

# external packages
import pandas as pd
from subgrounds.pagination import ShallowStrategy
from subgrounds import Subgrounds

logging.basicConfig(level=logging.INFO)


class Subgraph:
    def __init__(
        self,
        subgraph_url: str,
    ):

        self.sg = Subgrounds()

        try:
            self.subgraph = self.sg.load_subgraph(subgraph_url)
        except:
            raise ValueError("Invalid subgraph URL")


class GraphNetworkArbitrum(Subgraph):
    def __init__(
        self,
        subgraph_url="https://gateway.thegraph.com/api/{api_key}/subgraphs/id/DZz4kDTdmzWLWsV373w2bSmoar3umKKH9y82SUKr5qmp",
        api_key=None,
    ):
        if api_key is None:
            api_key = os.getenv("GRAPH_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key not found. Please provide one or set GRAPH_API_KEY in your environment or .env file."
                )

        formatted_subgraph_url = subgraph_url.format(api_key=api_key)

        super().__init__(formatted_subgraph_url)

    def build_subgraph_ipfs_query(self, subgraph_id: str):
        return self.subgraph.Query.subgraphs(
            where=[self.subgraph.Subgraph.id == subgraph_id]
        )

    def query_subgraph_ipfs(self, subgraph_id: str):
        query = self.build_subgraph_ipfs_query(subgraph_id)
        return self.sg.query_df(
            [
                query.metadata.codeRepository,
                query.currentVersion.subgraphDeployment.ipfsHash,
            ]
        )

    def get_subgraph_ipfs_hash(self, subgraph_id: str):
        ipfs_hash = self.query_subgraph_ipfs(subgraph_id).iloc[0, 1]
        return ipfs_hash

    def get_subgraph_ipfs_manifest(
        self,
        subgraph_id: str,
        ipfs_url: Optional[
            str
        ] = "https://api.thegraph.com/ipfs/api/v0/cat?arg={ipfs_hash}",
    ):
        ipfs_hash = self.get_subgraph_ipfs_hash(subgraph_id)
        formatted_ipfs_url = ipfs_url.format(ipfs_hash=ipfs_hash)
        response = requests.get(formatted_ipfs_url)

        if response.status_code == 200:
            return yaml.safe_load(response.text)
        else:
            raise ValueError(f"Failed to get IPFS manifest for subgraph {subgraph_id}.")

    def find_matching_values(
        self, data: Any, target_key: str = "abis"
    ) -> List[List[Dict]]:
        results = []

        def is_valid_abi_structure(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            return (
                "file" in item
                and "name" in item
                and isinstance(item["file"], dict)
                and "/" in item["file"]
                and isinstance(item["file"]["/"], str)
                and isinstance(item["name"], str)
            )

        def is_valid_abis_list(value: Any) -> bool:
            if not isinstance(value, list):
                return False
            return all(is_valid_abi_structure(item) for item in value)

        def search_recursive(current_data: Any):
            if isinstance(current_data, dict):
                for key, value in current_data.items():
                    if key == target_key and is_valid_abis_list(value):
                        results.append(value)
                    elif isinstance(value, (dict, list)):
                        search_recursive(value)

            elif isinstance(current_data, list):
                for item in current_data:
                    if isinstance(item, (dict, list)):
                        search_recursive(item)

        search_recursive(data)
        return results

    def flatten_list_of_unknown_depth(self, nested_list):
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(self.flatten_list_of_unknown_depth(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def get_abis_hashes_from_manifest(
        self,
        subgraph_id: str,
        ipfs_url: Optional[
            str
        ] = "https://api.thegraph.com/ipfs/api/v0/cat?arg={ipfs_hash}",
    ):
        manifest = self.get_subgraph_ipfs_manifest(subgraph_id, ipfs_url)
        return self.find_matching_values(manifest)

    def get_abis_from_manifest(
        self,
        subgraph_id: str,
        ipfs_url: Optional[
            str
        ] = "https://api.thegraph.com/ipfs/api/v0/cat?arg={ipfs_hash}",
    ):
        abis_hashes = self.get_abis_hashes_from_manifest(subgraph_id, ipfs_url)

        abis = {}
        for abi in abis_hashes:
            abi = abi[0]
            logging.info(abi)
            path = abi["file"]["/"].removeprefix("/ipfs/")
            name = abi["name"]

            if f"{name}-{path}" in abis:
                continue

            response = requests.get(ipfs_url.format(ipfs_hash=path))
            logging.info("Trying to get ABI for %s. Path: %s", name, path)
            if response.status_code == 200:
                abis[f"{name}-{path}"] = json.loads(response.text)
            else:
                raise ValueError(f"Failed to get ABI for {name}.")
        return abis

    def send_graphql_request(self, endpoint, query):
        headers = {"Content-Type": "application/json"}

        payload = {"query": query, "operationName": "Subgraphs", "variables": {}}

        response = requests.post(endpoint, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def single_query_to_df(self, endpoint, query):
        response = self.send_graphql_request(endpoint, query)
        keys = list(response["data"].keys())
        return pd.DataFrame(response["data"][keys[0]])
