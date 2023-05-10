import html
import itertools
from os import PathLike
import networkx as nx


def create_label(content: str):
    return f'<{content.replace(":", "&#58;")}>'


def get_graph_drawable(graph: nx.DiGraph) -> nx.DiGraph:
    graph = graph.copy()  # type: ignore
    degree = nx.degree(graph)

    for node in graph.nodes:
        handle = graph.nodes[node]

        handle["label"] = create_label(
            f'<b>{node}</b><br/>{html.escape(handle["instr"])}'
        )
        if degree[node] == 0:
            handle["color"] = "gray"  # type: ignore
        handle["shape"] = "box"

        del handle["instr"]

    for u, v in itertools.pairwise(range(len(graph.nodes))):
        if (u, v) not in graph.edges:
            graph.add_edge(u, v, color="gray", arrowhead="none", style="dotted")

    return graph


def draw(graph: nx.DiGraph, path: PathLike | str):
    a = nx.drawing.nx_agraph.to_agraph(get_graph_drawable(graph))
    a.node_attr["fontname"] = "Iosevka"
    a.write(path)
