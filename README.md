<!--
SPDX-FileCopyrightText: 2025 Semiotic AI, Inc.

SPDX-License-Identifier: Apache-2.0
-->

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# graphdoc
subgraph documentation generation

## Categorization

We adopt the methodology of Wretblad et al. [1] in defining both the difficulty of documenting a tables column and the quality of a column's description. 

### Column Difficulty Categorization

| Difficulty Level | Description                                                                                                                                                      |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Very Hard        | Given the database name, the table name, the column name, example data from the database, and other columns in the table, it is impossible to accurately determine what the column description should be. |
| Hard             | Given the database name, the table name, the column name, example data from the database, and other columns in the table, I am unsure what the column description should be.                             |
| Medium           | Given the database name, the table name, the column name, example data from the database, and other columns in the table, I can accurately determine what the column description should be.              |
| Easy             | Given only the table name and the column name, and other columns in the table, I can accurately determine what the column description should be.                                                       |

### Column Description Categorization (Gold)

| Classification    | Description                                                                                                                                                                                                                                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Perfect           | A perfect column description should contain enough information so that the interpretation of the column is completely free of ambiguity. It does not need to include any descriptions of the specific values inside the column to be considered perfect. The description should contain information about what table the column is referencing. For example, instead of "The name," we want "The name of the client that made the transaction" if we have a transaction database with columns such as NAME, AMOUNT, and DATE to resolve the ambiguity of what the name refers to. Additionally, the column description should be a full and valid English sentence, with proper grammar, capitalization, and punctuation. For instance, instead of "nationality of drivers" when each instance refers to only one driver, it should be "The nationality of a driver." |
| Poor but Correct  | The column description is poor but correct, but there is room for improvement.                                                                                                                                                                                                                              |
| Incorrect         | The column description is incorrect. Contains inaccurate or misleading information. It could still contain correct information, but any incorrect information automatically leads to an incorrect rating.                                                                                                     |
| No Description    | The column description is missing.                                                                                                                                                                                                                                                                         |
| I Can’t Tell      | It is impossible to tell the class of the description with the given information.                                                                                                                                                                                                                           |

### Column Description Categorization (Generated)

| Quality Level     | Description                                                                                                                                                                                                                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Perfect           | Matching the gold description without extra, redundant information. Redundant information is categorized as descriptions that do not provide useful additional information. For example, "<Gold description> + ‘is a primary/foreign key’" (NOT REDUNDANT) versus "<Gold description> + ‘is useful for retrieving data’" (REDUNDANT). |
| Almost Perfect    | Matching the gold description but verbose with redundant information, without any incorrect or misleading information.                                                                                                                                                                           |
| Poor but Correct  | The column description is poor but correct but has room for improvement due to missing information. For example, "The Time column records the specific time at which a transaction occurred, formatted in a 24-hour HH:MM pattern," which lacks enough information to make a valid prediction beyond the primary purpose.               |
| Incorrect         | The column description is incorrect and contains inaccurate or misleading information. Any incorrect information automatically leads to an incorrect rating, even if some correct information is present.                                                                                         |


## Subgraphs 

[arbitrum-one-bridge-explorer]: https://thegraph.com/explorer/subgraphs/6XazDBEjAVADSXbiBoXBBVwxTYf4PXRtucxn5vRQFLch
[arbitrum-one-bridge-github]: https://github.com/messari/subgraphs/tree/master/subgraphs/arbitrum-one-bridge

[gmx-forks-explorer]: https://thegraph.com/explorer/subgraphs/DiR5cWwB3pwXXQWWdus7fDLR2mnFRQLiBFsVmHAH9VAs
[gmx-forks-github]: https://github.com/messari/subgraphs/tree/master/subgraphs/gmx-forks

[uniswap-v3-forks-explorer]: https://thegraph.com/explorer/subgraphs/FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX
[uniswap-v3-forks-github]: https://github.com/messari/subgraphs/blob/master/subgraphs/uniswap-v3-forks

[bancor-v3-explorer]: https://thegraph.com/explorer/subgraphs/4Q4eEMDBjYM8JGsvnWCafFB5wCu6XntmsgxsxwYSnMib
[bancor-v3-github]: https://github.com/messari/subgraphs/blob/master/subgraphs/bancor-v3

[aave-forks-explorer]: https://thegraph.com/explorer/subgraphs/C2zniPn45RnLDGzVeGZCx2Sw3GXrbc9gL4ZfL8B8Em2j
[aave-forks-github]: https://github.com/messari/subgraphs/tree/master/subgraphs/aave-forks

[opensea-explorer]: https://thegraph.com/explorer/subgraphs/ECtdoov16DUmk5qbhFx4PVVN7vidiNDwzFNsui6FoHEo
[opensea-github]: https://github.com/messari/subgraphs/tree/master/subgraphs/opensea

[arrakis-finance-explorer]: https://thegraph.com/explorer/subgraphs/6yqMWioX8XNx2aMDYJGnvrVQWNrZfgBzY3ee1RmkXh5Z
[arrakis-finance-github]: https://github.com/messari/subgraphs/tree/master/subgraphs/arrakis-finance

[eigenlayer-explorer]: https://thegraph.com/explorer/subgraphs/68g9WSC4QTUJmMpuSbgLNENrcYha4mPmXhWGCoupM7kB
[eigenlayer-github]: https://github.com/messari/subgraphs/blob/master/subgraphs/eigenlayer

[livepeer-explorer]: https://thegraph.com/explorer/subgraphs/FE63YgkzcpVocxdCEyEYbvjYqEf2kb1A6daMYRxmejYC
[livepeer-github]: https://github.com/livepeer/subgraph/tree/main

[ens-subgraph-explorer]: https://thegraph.com/explorer/subgraphs/9sVPwghMnW4XkFTJV7T53EtmZ2JdmttuT5sRQe6DXhrq
[ens-subgraph-github]: https://github.com/ensdomains/ens-subgraph/tree/master

[graph-network-arbitrum-explorer]: https://thegraph.com/explorer/subgraphs/DZz4kDTdmzWLWsV373w2bSmoar3umKKH9y82SUKr5qmp
[graph-network-arbitrum-github]: https://github.com/graphprotocol/graph-network-subgraph

[known-origin-explorer]: https://thegraph.com/explorer/subgraphs/3VLK3AAxZSsysAC6SnF6BgFJ68SA5pRL1zGonvw2F2BT
[known-origin-github]: https://github.com/knownorigin/known-origin-subgraph

| Name                   | Explorer                                  | Github                              | Creator      | Type                                    |
|------------------------|-------------------------------------------|-------------------------------------|--------------|-----------------------------------------|
| arbitrum-one-bridge    | [link][arbitrum-one-bridge-explorer]    | [link][arbitrum-one-bridge-github]    | messari      | messari: schema-bridge                  |
| gmx-forks              | [link][gmx-forks-explorer]              | [link][gmx-forks-github]              | messari      | messari: schema-derivatives-perpfutures |
| uniswap-v3-forks       | [link][uniswap-v3-forks-explorer]       | [link][uniswap-v3-forks-github]       | messari      | messari: schema-dex-amm-extended        |
| bancor-v3              | [link][bancor-v3-explorer]              | [link][bancor-v3-github]              | messari      | messari: schema-dex-amm-extended        |
| aave-forks             | [link][aave-forks-explorer]             | [link][aave-forks-github]             | messari      | messari: schema-lending                 |
| opensea                | [link][opensea-explorer]                | [link][opensea-github]                | messari      | messari: schema-nft-marketplace         |
| arrakis-finance        | [link][arrakis-finance-explorer]        | [link][arrakis-finance-github]        | messari      | messari: schema-yield                   |
| eigenlayer             | [link][eigenlayer-explorer]             | [link][eigenlayer-github]             | messari      | messari: schema-non-standard            |
| livepeer               | [link][livepeer-explorer]               | [link][livepeer-github]               | livepeer     | livepeer: main                          |
| ens-subgraph           | [link][ens-subgraph-explorer]           | [link][ens-subgraph-github]           | ens          | ens: main                               |
| graph-network-arbitrum | [link][graph-network-arbitrum-explorer] | [link][graph-network-arbitrum-github] | e&n          | graph: network arbitrum                 |
| known-origin           | [link][known-origin-explorer]           | [link][known-origin-github]           | known-origin | known-origin                            |

## References

1. Wretblad, Niklas et al. *Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance*. arXiv preprint arXiv:2408.04691, 2024.
