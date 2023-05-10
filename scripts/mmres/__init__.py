import pandas as pd
import networkx as nx

from hashlib import md5

def get_dataframe():
    df_annotation = pd.read_json(
        "../data/mm-res/annotation.json",
        dtype={
            "user": "string",
            "recipe_id": "string",
            "labeled_edges": "object",
        },  # type: ignore
    ).rename(
        {
            "recipe_id": "id",
            "labeled_edges": "edges",
        },
        axis=1,
    )

    df_dataset = pd.read_json(
        "../data/mm-res/dataset.json",
        dtype={
            "ID": "string",
            "source": "string",
            "steps": "object",
            "images": "object",
        },  # type: ignore
    ).rename(
        {
            "ID": "id",
        },
        axis=1,
    )

    df = df_dataset.drop("images", axis=1).join(
        df_annotation.drop("user", axis=1).set_index("id"),
        on="id",
    )

    return df

def get_graph(row: pd.Series) -> nx.DiGraph:
    instrs = get_instrs_from_steps(row["steps"])
    graph = nx.DiGraph()

    graph.add_nodes_from((idx, dict(index=idx, instr=instr)) for idx, instr in enumerate(instrs))
    graph.add_edges_from(row["edges"])

    return graph

def get_instrs_from_steps(steps) -> list[str]:
    return [step["text"].strip() for step in steps]

if __name__ == "__main__":
    df = get_dataframe()
    df_dataset = pd.read_json(
            "../data/mm-res/dataset.json",
            dtype={
                "ID": "string",
                "source": "string",
                "steps": "object",
                "images": "object",
            },  # type: ignore
        ).rename(
            {
                "ID": "id",
            },
            axis=1,
        )

    graphs = df.apply(mmres.get_graph, axis=1) # type: ignore


    import draw
    draw.draw(graphs[0], "mmres.dot")
    # !dot -T png mmres.dot > mmres.png


    df["instrs"] = df["steps"].apply(get_instrs_from_steps)
    df.drop(["steps"], axis=1).reset_index(drop=True).to_json("../data/mm-res.json")
