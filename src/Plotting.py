# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src import Equilibirium as eq
from src import TAPOptimization as tap

np.set_printoptions(precision=3, suppress=True)


def set_size(width="default", fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "default":
        width_pt = 510 * 2
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def mpl_params(fontsize=14):
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "sans-serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.titlesize": fontsize,
        "figure.titlesize": fontsize,
        "axes.labelsize": fontsize,  # LaTeX default is 10pt font.
        "font.size": fontsize,
        "legend.fontsize": fontsize,  # Make the legend/label fonts
        "xtick.labelsize": fontsize,  # a little smaller
        "ytick.labelsize": fontsize,
        "figure.figsize": set_size(),  # default fig size of 0.9 textwidth
        "pgf.preamble": "\n".join(
            [  # plots will use this preamble
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
            ]
        ),
        "figure.constrained_layout.use": True,  # set constrained_layout to True
    }

    mpl.rcParams.update(pgf_with_latex)


def graphPlot(
    graph,
    ax=None,
    ec=None,
    nc=None,
    norm="Normalize",
    cmap=plt.get_cmap("cividis"),
    edge_labels=None,
    cbar=True,
    show_labels=False,
    title="default",
    edgewith=2,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # tt_func = nx.get_edge_attributes(graph, "tt_function")
    # weights = nx.get_edge_attributes(graph, "weight")
    pos = nx.get_node_attributes(graph, "pos")

    if ec is None:
        ec = np.ones(graph.number_of_edges())

    if isinstance(ec, str):
        flows = {nx.get_edge_attributes(graph, ec)}
    elif isinstance(ec, list):
        flows = dict(zip(graph.edges(), ec))
    elif isinstance(ec, np.ndarray):
        flows = dict(zip(graph.edges(), ec))
    elif isinstance(ec, dict):
        flows = ec
    # elif cc

    vmax = max(flows.values())
    vmin = min(flows.values())
    if norm == "Normalize":
        # cmap = plt.get_cmap("cividis")
        cmap.set_under("lightgrey")
        cmap.set_bad("lightgrey")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if norm == "LogNorm":
        vmin1 = vmax / 1000
        vmin = max(vmin, vmin1)
        cmap = plt.get_cmap("viridis")
        cmap.set_under("lightgrey")
        cmap.set_bad("lightgrey")
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm == "SymLogNorm":
        linthresh = max(1e-1, min(np.abs(list(flows.values()))))
        cmap = plt.get_cmap("coolwarm")
        norm = mpl.colors.SymLogNorm(
            linthresh=linthresh,
            # linscale=0.1,
            vmin=vmin,
            vmax=vmax,
        )

    edge_colors = {e: cmap(norm(flows[e])) for e in graph.edges()}
    if nc is None:
        node_colors = nx.get_node_attributes(graph, "color")
    elif not isinstance(nc, dict):
        if isinstance(nc, str):
            node_colors = nx.get_node_attributes(graph, nc)
        elif isinstance(nc, list) or isinstance(nc, np.ndarray):
            node_colors = dict(zip(graph.nodes(), nc))

        cmap = plt.get_cmap("viridis")
        norm = mpl.colors.Normalize(
            vmin=min(node_colors.values()), vmax=max(node_colors.values())
        )
        node_colors = {n: cmap(norm(node_colors[n])) for n in graph.nodes()}
    else:
        node_colors = nc

    if len(node_colors) == 0:
        node_colors = {n: "lightgrey" for n in graph.nodes()}

    if len(graph.nodes) < 25 or show_labels:
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_color=list(node_colors.values()), **kwargs
        )
        nx.draw_networkx_labels(graph, pos, ax=ax)
    else:
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_color="lightgrey", node_size=0, **kwargs
        )

    if nx.is_directed(graph):
        edges = graph.edges()
        for u, v in edges:
            # Draw with curvature if bidirectional
            if (v, u) in graph.edges():
                connection_style = "arc3,rad=0.2"
                label_pos = 0.5
            else:
                connection_style = "arc3,rad=0.0"
                label_pos = 0.45
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                connectionstyle=connection_style,
                edge_color=edge_colors[(u, v)],
                width=edgewith,
            )
            if edge_labels is not None:
                sublabels = {(u, v): np.round(edge_labels[(u, v)], 2)}
                nx.draw_networkx_edge_labels(
                    graph,
                    pos,
                    ax=ax,
                    edge_labels=sublabels,
                    connectionstyle=connection_style,
                    font_size=16,
                    font_family="sans-serif",
                    label_pos=label_pos,
                )
    else:
        # Draw straight lines if not bidirectional
        for u, v in graph.edges():
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                connectionstyle="arc3,rad=0.2",
                edge_color=edge_colors[(u, v)],
                width=edgewith,
            )
            if edge_labels is not None:
                sublabels = {(u, v): np.round(edge_labels[(u, v)], 1)}
                nx.draw_networkx_edge_labels(
                    graph,
                    pos,
                    ax=ax,
                    edge_labels=sublabels,
                    connectionstyle="arc3,rad=0.2",
                    font_size=12,
                    label_pos=0.45,
                )
    if cbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            pad=0.0,
            aspect=20,
            shrink=1 / 2,
            extend="min",
        )
        cbar.ax.set_title(r"$F_{i \rightarrow j}$")
    try:
        if title == "default":
            sc = int(tap.social_cost(graph, np.array(list(flows.values()))))
            ax.set_title(rf"$\sum_{{e\in E}} C(f_e)=${sc}", fontsize=12)
            pe = int(tap.potential_energy(graph, np.array(list(flows.values()))))
            ax.set_title(
                ax.get_title() + "\n" + rf"$\sum_{{e\in E}} U(f_e)=${pe}", fontsize=12
            )
    except:
        pass

    ax.axis("off")

    return ax


# %%
from sys import platform

if platform != "linux":
    if __name__ == "__main__":
        from src import Equilibirium as eq

        g = nx.erdos_renyi_graph(10, 0.3, directed=True, seed=42)
        pos = nx.spring_layout(g)
        nx.set_node_attributes(g, pos, "pos")
        nx.set_edge_attributes(g, 1, "weight")
        P = np.zeros(g.number_of_nodes())
        P[0] = 1
        P[-1] = -1
        nx.set_node_attributes(g, dict(zip(g.nodes, P)), "P")

        F = eq.linear_flow(g)
        Fvec = np.array(list(F.values()))
        graphPlot(g, cc=Fvec, edge_labels=F)

    mpl_params(fontsize=14)

# %%

# %%
