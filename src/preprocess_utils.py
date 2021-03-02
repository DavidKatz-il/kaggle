import pandas as pd
import networkx as nx
from itertools import combinations
from typing import Callable, Union, List
import random
from functools import reduce


def interactions(
    df: pd.DataFrame, calcs: Union[Callable, List[Callable]], r: int = 2
) -> pd.DataFrame:
    if not isinstance(calcs, list):
        calcs = [calcs]
    for comb in combinations(df.columns, r):
        for calc in calcs:
            df[f"_{calc.__name__}_".join(comb)] = reduce(calc, [df[c] for c in comb])
    return df


def correlated_cols(df: pd.DataFrame, treshold: int) -> list:
    corrMatrix = df.corr()
    corrMatrix.loc[:, :] = abs(np.tril(corrMatrix, k=-1))
    _corrMatrix = (
        corrMatrix[corrMatrix > treshold]
        .stack()
        .reset_index()
        .rename(columns={"level_0": "source", "level_1": "target", 0: "weight"})
    )
    G = nx.from_pandas_edgelist(
        _corrMatrix, edge_attr=["weight"], create_using=nx.Graph()
    )
    sub_graphs = [list(G.subgraph(c).nodes()) for c in nx.connected_components(G)]
    to_drop = []
    for sub_graph in sub_graphs:
        sub_graph.remove(random.choice(sub_graph))
        to_drop.extend(sub_graph)
    return to_drop


def cols_to_drop_base_on_pval(df: pd.DataFrame, threshold: int) -> list:
    p_val_cols_todrop = []
    for col in df.columns:
        if col == "target":
            continue
        r, p_val = pearsonr(df[col], df["target"])
        if p_val > threshold:
            p_val_cols_todrop.append(col)
    return p_val_cols_todrop


def cols_pval(df: pd.DataFrame, threshold: int) -> dict:
    p_val_cols = {}
    for col in df.columns:
        if col == "target":
            continue
        r, p_val = pearsonr(df[col], df["target"])
        p_val_cols[col] = p_val
    return p_val_cols
