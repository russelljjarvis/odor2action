"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)

"""
from community import community_louvain

import igraph as ig
import plotly.graph_objs as go

from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

from hiveplotlib import Axis, Node, HivePlot, hive_plot_n_axes
from hiveplotlib.viz import hive_plot_viz_mpl

import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, output_file
import copy

import argparse
import numpy as np
import networkx as nx
import dash_bio as dashbio
import streamlit as st

#st.set_page_config(layout="wide")

import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import shelve
#import streamlit as st
import os
import pandas as pd
import pickle
import streamlit as st

# from holoviews import opts, dim
from collections import Iterable
import networkx

# import holoviews as hv
import chord2
import shelve

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np
import pickle

import plotly.graph_objects as go


import pandas as pd
import openpyxl
from pathlib import Path
import numpy as np
import networkx as nx

import xlrd
import matplotlib.pyplot as plt


import dash_bio


def disable_logo(plot, element):
    plot.state.toolbar.logo = None


# hv.extension("bokeh", logo=False)
# hv.output(size=150)
# hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)
import plotly.graph_objects as go

# from auxillary_methods import plotly_sized2

from datashader.bundling import hammer_bundle

from typing import List
import pandas as pd

# import holoviews as hv
import seaborn as sns


def generate_sankey_figure(
    nodes_list: List, edges_df: pd.DataFrame, title: str = "Sankey Diagram"
):

    edges_df["src"] = edges_df["src"].apply(lambda x: nodes_list.index(x))
    edges_df["tgt"] = edges_df["tgt"].apply(lambda x: nodes_list.index(x))
    # creating the sankey diagram
    data = dict(
        type="sankey",
        node=dict(
            hoverinfo="all",
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_list,
        ),
        link=dict(
            source=edges_df["src"], target=edges_df["tgt"], value=edges_df["weight"]
        ),
    )

    layout = dict(title=title, font=dict(size=10))

    fig = go.Figure(data=[data], layout=layout)
    st.write(fig)

    # return fig


# @st.cache
# @st.cache(suppress_st_warning=True)
def data_shade(graph, color_code, adj_mat, color_dict, labels_=False):

    nodes = graph.nodes
    # orig_pos=nx.get_node_attributes(graph,'pos')

    nodes_ind = [i for i in range(0, len(graph.nodes()))]
    redo = {k: v for k, v in zip(graph.nodes, nodes_ind)}
    # pos = nx.spring_layout(H, k=0.05, seed=4572321, scale=1)

    pos_ = nx.spring_layout(graph, scale=2.5, k=0.00015, seed=4572321)
    # node_color = [community_index[n] for n in graph]
    H = graph.to_undirected()
    centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
    node_size = [v * 25000 for v in centrality.values()]

    coords = []
    for node in graph.nodes:
        x, y = pos_[node]
        coords.append((x, y))
    nodes_py = [
        [new_name, pos[0], pos[1]]
        for name, pos, new_name in zip(nodes, coords, nodes_ind)
    ]
    ds_nodes = pd.DataFrame(nodes_py, columns=["name", "x", "y"])
    ds_edges_py = []
    for (n0, n1) in graph.edges:
        ds_edges_py.append([redo[n0], redo[n1]])
    ds_edges = pd.DataFrame(ds_edges_py, columns=["source", "target"])
    hb = hammer_bundle(ds_nodes, ds_edges)
    hbnp = hb.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0]
    start = 0
    segments = []
    # st.markdown(
    #    'It is also a bit like how parallel neurons are wrapped together closely by a density of myline cells, like neurons traveling through the corpus callosum'
    # )
    # st.markdown(
    #    "Think of it conceptually like Ramon Y Cajal principle of wiring cost optimization."
    # )
    # st.markdown(
    #    "Neurons processes projecting to very close places shouldnt travel down indipendant dedicated lines, there is less metabolic cost involved in \n channeling parallel fibres in the same myline sheath through a backbone like the corpus callosum."
    # )

    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        start = stop

    fig, ax = plt.subplots(figsize=(15, 15))
    widths = list(adj_mat["weight"].values)
    srcs = list(adj_mat["src"].values)

    for ind, seg in enumerate(segments):
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            c=color_code[srcs[ind]],
            alpha=0.67,
            linewidth=widths[ind],
        )
    node_color = [color_code[n] for n in graph]

    ax3 = nx.draw_networkx_nodes(
        graph,
        pos_,
        node_color=node_color,
        node_size=node_size,
        node_shape="o",
        alpha=0.5,
        vmin=None,
        vmax=None,
        linewidths=2.0,
        label=None,
        ax=ax,
    )  # , **kwds)

    axx = plt.gca()  # to get the current axis

    axx.collections[0].set_edgecolor("#FF0000")
    labels = {}
    for node in graph.nodes():
        # set the node name as the key and the label as its value
        labels[node] = node
    if labels_:
        nx.draw_networkx_labels(graph, pos_, labels, font_size=16, font_color="r")

    # ax3.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    for k, v in color_dict.items():
        plt.scatter([], [], c=v, label=k)
    plt.legend(frameon=False,prop={'size':24})

    def dontdo(segments, pos_, graph):
        fig.show()
        df_geo = {}
        df_geo["text"] = list(node for node in graph.nodes)

        fig = go.Figure()
        lats = []
        lons = []
        traces = []
        other_traces = []
        # from tqdm import tqdm
        for ind, seg in enumerate(segments):
            x0, y0 = seg[1, 0], seg[1, 1]  # graph.nodes[edge[0]]['pos']
            x1, y1 = seg[-1, 0], seg[-1, 1]  # graph.nodes[edge[1]]['pos']
            xx = seg[:, 0]
            yy = seg[:, 1]
            lats.append(xx)
            lons.append(yy)
            for i, j in enumerate(xx):
                if i > 0:
                    other_traces.append(
                        go.Scatter(
                            lon=[xx[i], xx[i - 1]],
                            lat=[yy[i], yy[i - 1]],
                            mode="lines",
                            showlegend=False,
                            hoverinfo="skip",
                            line=dict(width=0.5, color="blue"),
                        )
                    )
        fig.add_traces(other_traces)
        fig.add_trace(
            go.Scatter(
                lat=pos_,
                lon=pos_,
                marker=dict(
                    size=3,  # data['Confirmed-ref'],
                    color=colors,
                    opacity=1,
                ),
                text=list(graph.nodes),
                hovertemplate="%{text} <extra></extra>",
            )
        )
        fig["layout"]["width"] = 1825
        fig["layout"]["height"] = 1825
        st.write(fig)

    return fig  # ,colors

    # return fig

def depricated():
    def plot_stuff(df2, edges_df_full, first, adj_mat_dicts):
        with shelve.open("fast_graphs_splash.p") as db:
            flag = "chord" in db
            if False:  # flag:
                graph = db["graph"]

            else:
                db.close()


from hiveplotlib import Axis, Node, HivePlot
from hiveplotlib.viz import axes_viz_mpl, node_viz_mpl, edge_viz_mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


class renamer:
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])



def dontdo():
    df2.rename(columns=renamer(), inplace=True)

    df4 = pd.DataFrame()

    for col in df2.columns[0 : int(len(df2.columns) / 2)]:
        # if
        if col + str("_1") in df2.columns:
            df4[col] = df2[col] + df2[col + str("_1")]
        else:
            df4[col] = df2[col]

def dontdo():
    """
    for col in df2.columns:

                    if col.split("_")[0] in df3.columns and col.split("_")[0] in df5.columns:
                                    df4[col] = df2[col] + df3[col]
                                    #st.text('yes')

                    else:
                                    df4[col] = df2[col]
    """
import copy

# @st.cache(persist=True)
#@st.cache(allow_output_mutation=True)
def get_frame(threshold=6):

    with shelve.open("fast_graphs_splash.p") as store:
        flag = "df" in store
        if False:
            df = store["df"]  # load it

            df2 = store["df2"]  # load it
            names = store["names"]  # = names  # save it
            ratercodes = store["ratercodes"]  # =   # save it
            legend = store["legend"]  # = legend  # save it

        else:
            hard_codes = Path("code_by_IRG.xlsx")
            hard_codes = openpyxl.load_workbook(hard_codes)

            hard_codes = hard_codes.active

            hard_codes = pd.DataFrame(hard_codes.values)

            xlsx_file0 = Path("o2anetmap2021.xlsx")
            xlsx_file1 = Path("o2anetmap.xlsx")
            wb_obj0 = openpyxl.load_workbook(xlsx_file0)
            wb_obj1 = openpyxl.load_workbook(xlsx_file1)

            # Read the active sheet:
            worksheet0 = wb_obj0.active
            worksheet1 = wb_obj1.active

            df3 = pd.DataFrame(worksheet0.values)
            df2 = pd.DataFrame(worksheet1.values)

            df2 = pd.concat([df3, df2])
            sheet = copy.copy(df2)
            hc = {k:str("IRG ")+str(v) for k,v in zip(hard_codes[0][1::],hard_codes[1][1::])}
            hc1 = {k:"DCMT" for k,v in hc.items() if v=="IRG DCMT"}
            #st.text(hc1)
            hc.update(hc1)
            hc.pop("Code",None)

            #st.text(hc)
            color_code_0 = {
                k: v for k, v in zip(df2[0], df2[1]) if k not in "Rater Code"
            }
            #st.text(hc)
            #st.text(color_code_0)
            color_code_0.update(hc)

            #st.write(color_code_0)



            #for i, (node_id, degree) in enumerate(zip(node_ids, degrees)):
            #    if not reverse[node_id] in color_code_0.keys():
            #        color_code_0[reverse[node_id]] = hc[reverse[node_id]]
            #        reverse[node_id] = hc[reverse[node_id]]
            #➜  ~ change yellow to red
            #➜  ~ change orange to purple

            # Ribbon color code needs to labeled as to or from.
            # source or target.

            color_dict = {
                "IRG 3": "green",
                "IRG 1": "blue",
                "IRG 2": "red",
                "DCMT": "purple",
            }
            color_code_1 = {}

            popg = nx.DiGraph()

            for k, v in color_code_0.items():

                if v not in popg.nodes:
                    popg.add_node(v, name=v)
                color_code_1[k] = color_dict[v]
            col_to_rename = df2.columns
            ratercodes = df2[0][1::]
            row_names = list(df2.T[0].values)
            row_names.append(list(df2.T[0].values)[-1])
            row_names = row_names[2::]
            names = [rn[0].split("- ") for rn in row_names]
            names2 = []
            for i in names:
                if len(i) == 2:
                    names2.append(i[1])
                else:
                    names2.append(i)
            names = names2
            for nm in names:
                if nm not in color_code_1.keys():
                    color_code_1[nm] = "black"

            row_names = list(range(0, len(df2.columns) + 1, 1))
            to_rename = {k: v for k, v in zip(row_names, names)}
            r_names = list(df2.index.values[:])

            to_rename_ind = {v: k for k, v in zip(df2[0], r_names)}
            del df2[0]
            del df2[1]
            # del df2[112]
            del df2[113]
            df2.drop(0, inplace=True)
            df2.drop(1, inplace=True)


            df2.rename(columns=to_rename, inplace=True)
            df2.rename(index=to_rename_ind, inplace=True)
            unk = []

            for col in df2.columns:
                if col in df2.index.values[:]:
                    pass
                else:
                    pass
                    #st.text('found')
                    #st.text(hc[col])
                    #st.text(col)


            legend = {}

            legend.update({"Never": 0.0})
            legend.update({"Barely or never": 1})
            legend.update({"Occasionally in a minor way": 2})
            legend.update({"Less than once a month": 3})
            legend.update({"More than once a month (But not weekly)": 4})
            legend.update({"Occasionally but substantively": 5})
            legend.update({"More than twice a week": 6})
            legend.update({"Often": 7})
            legend.update({"Much or all of the time": 8})
            legend.update({"1-2 times a week": 9.0})
            df2.replace({"": 0.0}, inplace=True)
            df2.replace({" ": 0.0}, inplace=True)
            df2.replace({"\t": 0.0}, inplace=True)
            df2.replace({"\n": 0.0}, inplace=True)

            df2.replace({"Never": 0.0}, inplace=True)
            df2.replace({"Barely or never": 1}, inplace=True)
            df2.replace({"Occasionally in a minor way": 2}, inplace=True)
            df2.replace({"Less than once a month": 3}, inplace=True)
            df2.replace({"More than once a month (But not weekly)": 4}, inplace=True)
            df2.replace({"Occasionally but substantively": 5}, inplace=True)
            df2.replace({"More than twice a week": 6}, inplace=True)
            df2.replace({"Often": 7}, inplace=True)
            df2.replace({"Much or all of the time": 8}, inplace=True)
            df2.replace({"1-2 times a week": 9.0}, inplace=True)

            # This sums columns under the same name
            df2.groupby(df2.columns, axis=1).sum()
            df2.groupby(level=0, axis=1).sum()
            # df2 = df4
            store["df2"] = df2  # save it
            # st.write(df2)
            store["names"] = names  # save it
            store["ratercodes"] = ratercodes  # save it
            store["legend"] = legend  # save it

    return (
        df2,
        names,
        ratercodes,
        legend,
        color_code_1,
        color_dict,
        color_code_0,
        sheet,
        popg,
        hc
    )


sns_colorscale = [
    [0.0, "#3f7f93"],  # cmap = sns.diverging_palette(220, 10, as_cmap = True)
    [0.071, "#5890a1"],
    [0.143, "#72a1b0"],
    [0.214, "#8cb3bf"],
    [0.286, "#a7c5cf"],
    [0.357, "#c0d6dd"],
    [0.429, "#dae8ec"],
    [0.5, "#f2f2f2"],
    [0.571, "#f7d7d9"],
    [0.643, "#f2bcc0"],
    [0.714, "#eda3a9"],
    [0.786, "#e8888f"],
    [0.857, "#e36e76"],
    [0.929, "#de535e"],
    [1.0, "#d93a46"],
]


# @st.cache
def df_to_plotly(df, log=False):
    return {"z": df.values.tolist(), "x": df.columns.tolist(), "y": df.index.tolist()}


# @st.cache
def plot_df_plotly(sleep_df):
    fig = go.Figure(data=go.Heatmap(df_to_plotly(sleep_df, log=True)))
    st.write(fig)


# @st.cache
def plot_imshow_plotly(sleep_df):

    heat = go.Heatmap(df_to_plotly(sleep_df), colorscale=sns_colorscale)
    title = "Adjacency Matrix"

    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=600,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    fig = go.Figure(data=[heat], layout=layout)

    st.write(fig)


def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(
        walks,
        size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        iter=args.iter,
    )
    model.save_word2vec_format(args.output)

    return


#@st.cache(persist=True)
def get_table_download_link_csv(df):
    import base64

    # csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    # b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href

def draw_network(G,pos,ax,widths,edge_colors,sg=None):

    for n in G.nodes:
        c=Circle(pos[n],radius=0.05,alpha=0.7)
        #ax.add_patch(c)
        G.nodes[n]['patch']=c
        x,y=pos[n]
    seen={}
    for n,(u,v,d) in enumerate(G.edges(data=True)):
        n1=G.nodes[u]['patch']
        n2=G.nodes[v]['patch']
        rad=0.1
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1
        alpha=0.5
        color='k'

        e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=10.0,
                            lw=widths[n],
                            alpha=alpha,
                            color=edge_colors[n])
        seen[(u,v)]=rad
        ax.add_patch(e)
    return e

#@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def population(cc, popg, color_dict):

    fig, ax = plt.subplots(figsize=(20, 15))

    pos = nx.spring_layout(popg, k=15, seed=4572321, scale=1.5)
    sizes = {}
    for k, v in cc.items():
        if v not in sizes.keys():
            sizes[v] = 1
        else:
            sizes[v] += 1
    temp = list([s * 1000 for s in sizes.values()])
    node_color = [color_dict[n] for n in popg]
    nx.draw_networkx_nodes(
        popg,
        pos=pos,
        node_color=node_color,  # = [color_code[n] for n in H],
        node_size=temp,
        alpha=0.6,
        linewidths=2,
    )

    widths = []  # [e["weight"] for e in popg.edges]
    # st.text(widths)
    edge_list = []
    edge_colors = []
    for e in popg.edges:
        edge_list.append((e[0], e[1]))
        edge_colors.append(color_dict[e[0]])

        ee = popg.get_edge_data(e[0], e[1])
        widths.append(ee["weight"] * 0.02)

    # nx.draw_networkx_edges(G, pos, edgelist=edgelist, arrowstyle="<|-", style="dashed")
    def dontdo():
        '''
        nx.draw_networkx_edges(
            popg,
            pos=pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            alpha=0.70,
            width=widths,
            arrowstyle="<|-"
        )
        '''

    ax=plt.gca()
    draw_network(popg,pos,ax,widths,edge_colors)
    ax.autoscale()
    plt.axis('equal')
    plt.axis('off')

    # labels = {v.name:v for v,v in popg.nodes}
    labels = {}
    for node in popg.nodes():
        # set the node name as the key and the label as its value
        labels[node] = node

    for k, v in labels.items():
        plt.scatter([], [], c=color_dict[v], label=k)
    plt.legend(frameon=False,prop={'size':34})

    #nx.draw_networkx_labels(popg, pos, labels, font_size=16, font_color="r")
    popgc = copy.copy(popg)
    #popgc.graph["edge"] = {"arrowsize": "0.6", "splines": "curved"}
    #popgc.graph["graph"] = {"scale": "3"}


    #st.markdown(""" Missing self connections, but node size proportions""")
    st.pyplot(fig)
    try:
        from networkx.drawing.nx_agraph import to_agraph

        dot = to_agraph(popgc)
        dot.layout("dot")
        st.markdown(""" Schematic View""")
        st.graphviz_chart(dot.to_string())
    except:
        pass
from scipy.spatial import ConvexHull, convex_hull_plot_2d


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def community_layout(g, partition):
    """
    Compute the layout for a modular graph.
    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot
    partition -- dict mapping int node -> int community
        graph partitions
    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions
    """
    pos_communities = _position_communities(g, partition, k=0.04, scale=5.)
    pos_nodes = _position_nodes(g, partition, k=0.04, scale=1.)
    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos, pos_communities

def _position_communities(g, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def colored_hive_axis(first,color_code_0,reverse):
    c = ['#e41a1c', '#377eb8', '#4daf4a',
         '#984ea3', '#ff7f00', '#ffff33',
         '#a65628', '#f781bf', '#999999',]
    IRG1_indices = []
    IRG2_indices = []
    IRG3_indices = []
    DCMT_indices = []  # ,Un_ind
    g = first
    forwards = {v:k for k,v in reverse.items()}
    for i, (node_id) in enumerate(g.nodes):
        if forwards[node_id] in color_code_0.keys():
            if color_code_0[forwards[node_id]] == "IRG 1":
                IRG1_indices.append(node_id)
            if color_code_0[forwards[node_id]] == "IRG 2":
                IRG2_indices.append(node_id)
            if color_code_0[forwards[node_id]] == "IRG 3":
                IRG3_indices.append(node_id)
            #if color_code_0[forwards[node_id]] == "DCMT":
            #    DCMT_indices.append(node_id)


    # create hiveplot object
    h = Hiveplot()


    '''
    fig = plt.figure()
    # create three axes, spaced at 120 degrees from each other

    h.axes = [Axis(start=20, angle=0,
                   stroke=random.choice(c), stroke_width=1.1),
              Axis(start=20, angle=90,
                   stroke=random.choice(c), stroke_width=1.1),
              Axis(start=20, angle=90 + 90,
                   stroke=random.choice(c), stroke_width=1.1)]


              #Axis(start=20, angle=90 + 90 + 90,
              #        stroke=random.choice(c), stroke_width=1.1)
              #]
    '''
    fig = plt.figure()
    # create three axes, spaced at 120 degrees from each other
    h.axes = [Axis(start=20, angle=0,
                   stroke=random.choice(c), stroke_width=1.1),
              Axis(start=20, angle=120,
                   stroke=random.choice(c), stroke_width=1.1),
              Axis(start=20, angle=120 + 120,
                   stroke=random.choice(c), stroke_width=1.1)
              ]

    #g = first

    # place these nodes into our three axes
    for axis, nodes in zip(h.axes,
                           [IRG1_indices, IRG2_indices, IRG3_indices]):
        circle_color = random.choice(c)
        for v in nodes:
            st.text(v)
            # create node object
            node = Node(radius=15,
                        label="node %s" % v)
            # add it to axis
            st.text(node)
            axis.add_node(v, node)
            # once it has x, y coordinates, add a circle
            node.add_circle(fill=circle_color, stroke=circle_color,
                            stroke_width=0.1, fill_opacity=0.7)
            if axis.angle < 180:
                orientation = -1
                scale = 0.6
            else:
                orientation = 1
                scale = 0.35
            # also add a label
            node.add_label("node %s" % (v),
                           angle=axis.angle + 90 * orientation,
                           scale=scale)

    # iterate through axes, from left to right
    for n in range(-1, len(h.axes) - 1):
        curve_color = random.choice(c)
        # draw curves between nodes connected by edges in network
        h.connect_axes(h.axes[n],
                       h.axes[n+1],
                       g.edges,
                       stroke_width=0.5,
                       stroke=curve_color)
    # save output
    h.save('col_ba_hiveplot1.svg')
    #from PIL import Image
    f = open('col_ba_hiveplot1.svg',"r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)


    #st.image(Image.open("col_ba_hiveplot.svg"))


#@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def community(first,color_code,color_dict):
    my_expander = st.beta_expander("Toggle node labels")
    labels_ = my_expander.radio("Would you like to label nodes?", ("Yes", "No"))
    if labels_ == "Yes":
        labelsx = True
    if labels_ == "No":
        labelsx = False

    from community import community_louvain
    temp = first.to_undirected()

    partition = community_louvain.best_partition(temp,resolution=4.0)
    pos, pos_communities = community_layout(temp, partition)
    diffcc = list(partition.values())
    pkeys = set(partition.values())
    partitiondf = pd.DataFrame([partition]).T
    hulls = []
    pointss = []
    whichpkeys = []
    centrex=[]
    centrey=[]

    for k in pkeys: # iterate over communities
        list_of_nodes = partitiondf[partitiondf.values==k].index.values[:]
        tocvxh = []
        for node in list_of_nodes:
            x,y = pos[node]
            tocvxh.append((x,y))

        meanx = np.mean([i[0] for i in tocvxh])
        meany = np.mean([i[1] for i in tocvxh])
        centrex.append(meanx)
        centrey.append(meany)

        if len(tocvxh) <=2:
            pass

        if len(tocvxh) >2:
            hull = ConvexHull(tocvxh)
            pointss.append(tocvxh)
            hulls.append(hull)
            whichpkeys.append(diffcc[k])

    #st.write(partitiondf)
    centrality = nx.betweenness_centrality(temp, k=10, endpoints=True)
    #edge_thickness = {k: v * 20000 for k, v in centrality.items()}
    node_size = [v * 10000 for v in centrality.values()]

    #diffccl = list(partition.ite())

    srcs = []
    widths = []


    for e in temp.edges:
        src = partition[e[0]]
        srcs.append(src)
        ee = temp.get_edge_data(e[0], e[1])
        widths.append(0.85*ee["weight"])
    fig1,ax = plt.subplots(figsize=(20,20))



    #for i,hull in enumerate(hulls):
    #    points = np.array(pointss[i])
    #    for simplex in hull.simplices:
            #pass
    #        plt.fill(points[hull.vertices,0], points[hull.vertices,1], color=str(whichpkeys[i]), alpha=0.05)


    nx.draw_networkx_nodes(
        temp,
        pos=pos,
        node_color=diffcc,
        node_size=550,
        alpha=0.5,
        linewidths=1,
    )

    axx = fig1.gca()  # to get the current axis
    axx.collections[0].set_edgecolor("#FF0000")
    label_pos = copy.deepcopy(pos)
    for k,v in label_pos.items():
        label_pos[k][0] = v[0]+0.5

    if labelsx:
        labels = {}
        for node in temp.nodes():
            # set the node name as the key and the label as its value
            labels[node] = node
        nx.draw_networkx_labels(temp, label_pos, labels, font_size=29.5, font_color="b")

    nx.draw_networkx_edges(
        temp, pos=pos, edge_color='grey', alpha=0.15, width=widths
    )
    import matplotlib.patches as patches
    for centre in zip(centrex,centrey):
        r = 1.25;
        c = (float(centre[0]),float(centre[1]))
        ax.add_patch(plt.Circle(c, r, color='#00ff33', alpha=0.15))
    plt.axis('off')

    plt.savefig("img1.png")


    fig2,ax = plt.subplots(figsize=(20,20))

    node_color = [color_code[n] for n in first]
    srcs = []
    for e in temp.edges:
        src = color_code[e[0]]
        srcs.append(src)

    nx.draw_networkx_nodes(
        temp,
        pos=pos,
        node_color=node_color,
        node_size=550,
        alpha=0.5,
        linewidths=1,
    )
    axx = fig2.gca()  # to get the current axis
    axx.collections[0].set_edgecolor("#FF0000")
    #if labelsx:
    #    nx.draw_networkx_labels(temp, label_pos, labels, font_size=9.5, font_color="b")

    #nx.draw(temp, pos, node_color=node_color)
    nx.draw_networkx_edges(
        temp, pos=pos, edge_color='grey', alpha=0.15, width=widths
    )
    for centre in zip(centrex,centrey):
        r = 1.5;
        c = (float(centre[0]),float(centre[1]))
        ax.add_patch(plt.Circle(c, r, color='#00ff33', alpha=0.15))

    for k, v in color_dict.items():
        plt.scatter([], [], c=v, label=k)
    plt.legend(frameon=False,prop={'size':29.5})
    plt.axis('off')
    plt.savefig("img2.png")
    import matplotlib.image as mpimg
    img1 = mpimg.imread('img1.png')
    img2 = mpimg.imread('img2.png')
    #@fig3 = plt.figure(1)
    #plt.subplot(211)
    #fig2,ax = plt.subplots()

    fig3, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,70))

    ax1.imshow(img1)
    ax1.axis('off')
    #plt.subplot(221)
    ax2.imshow(img2)
    ax2.axis('off')

    st.pyplot(fig3, use_column_width=True)
    #st.beta_set_page_config(layout="wide")
    #col1, col2 = st.beta_columns(2)
    #col1.pyplot(fig1, use_column_width=True)

    #col2.pyplot(fig2, use_column_width=True)

    #plt#.show()
#@st.cache(allow_output_mutation=True,suppress_st_warning=True)


def list_centrality(first):
    H = first.to_undirected()
    st.markdown("## Betweeness Centrality:")
    st.markdown("Top to bottom node id from most central to least:")

    centrality = nx.betweenness_centrality(H, endpoints=True)
    df = pd.DataFrame([centrality])
    df = df.T
    df.sort_values(0, axis=0, ascending=False, inplace=True)
    df.rename(columns={0:'centrality value'},inplace=True)

    bc = df
    st.markdown("### Most Connected:")
    st.write(bc.head())
    st.text("...")
    st.markdown("### Least Connected:")
    st.write(bc.tail())

    st.markdown("## In degree Centrality: (percieved listeners/high authority)")
    st.markdown("Top to bottom node id from most central to least:")

    centrality = nx.in_degree_centrality(first)
    df = pd.DataFrame([centrality])
    df = df.T
    df.sort_values(0, axis=0, ascending=False, inplace=True)
    df.rename(columns={0:'centrality value'},inplace=True)
    st.markdown("### Biggest Listeners:")

    st.write(df.head())
    st.text("...")
    st.markdown("### Least Listening:")

    st.write(df.tail())

    #bc = df
    #st.table(df)

    #Compute the in-degree centrality for nodes.
    st.markdown("## Out-degree Centrality (percieved talkers), read from top to bottom from most central to least:")

    centrality = nx.out_degree_centrality(first)
    df = pd.DataFrame([centrality])
    df = df.T
    df.sort_values(0, axis=0, ascending=False, inplace=True)
    df.rename(columns={0:'centrality value'},inplace=True)
    st.markdown("### Biggest Talkers:")

    st.write(df.head())
    st.text("...")
    st.markdown("### Least Talkative:")

    st.write(df.tail())

    #bc = df
    #st.table(df)
    return bc

    #Compute the in-degree centrality for nodes.
    #st.markdown("Out-degree Centrality:")
    #st.markdown("Top to bottom node id from most central to least:")

    #Compute the out-degree centrality for nodes.
    #st.markdown("Betweeness Centrality:")
    #centrality = nx.betweenness_centrality(H, endpoints=True)
    #df = pd.DataFrame([centrality])
    #df = df.T
    #df.sort_values(0, axis=0, ascending=False, inplace=True)
    #st.table(df)
    #edge_thickness = {k: v * 200000 for k, v in centrality.items()}

def physics(first, adj_mat_dicts, color_code,color_code_0,color_dict):

    my_expander = st.beta_expander("physical parameters")

    phys_ = my_expander.radio(
        "Would you like to change physical parameters?", ("No", "Yes")
    )
    pos = nx.get_node_attributes(first, "pos")
    # fig = plt.figure()
    d = nx.degree(first)
    temp = first.to_undirected()
    cen = nx.betweenness_centrality(temp)
    d = [((d[node] + 1) * 50000) for node in first.nodes()]
    G = first  # ead_graph()

    nt = Network(
        notebook=True,
        directed=True,
        height="500px",
        width="100%",
        font_color="black",  # , bgcolor='#222222'
    )  # bgcolor='#222222',

    nt = Network(
        "500px", "500px", notebook=True
    )

    nt.barnes_hut()
    nt.from_nx(G)

    adj_mat = pd.DataFrame(adj_mat_dicts)
    edge_data = zip(
        list(adj_mat["src"].values),
        list(adj_mat["tgt"].values),
        list(adj_mat["weight"].values),
    )

    H = first.to_undirected()
    centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
    edge_thickness = {k: v * 200000 for k, v in centrality.items()}
    node_size = {k: v * 200000 for k, v in centrality.items()}

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2] * 1350.0
        src = str(src)
        dst = str(dst)
        w = float(w)
        nt.add_edge(src, dst, value=w)

    neighbor_map = nt.get_adj_list()
    my_expander = st.beta_expander("Mouse over node info?")

    mo_ = my_expander.radio(
        "Toggle Mouse overs?", ("No", "Yes")
    )
    if mo_ == "Yes":
        mo = True
    else:
        mo = False
    #labels = False
    if phys_ == "Yes":
        nt.show_buttons(filter_=["physics"])

    # add neighbor data to node hover data
    for node in nt.nodes:
        if mo:
            if "title" not in node.keys():
                if node["id"] in color_code_0.keys():
                    node["title"] = "<br> This node is:"+str(node["id"])+"<br> it's membership is "+str(color_code_0[node["id"]])+" It's neighbors are:<br>" + "<br>".join(neighbor_map[node["id"]])
                else:
                    node["title"] = "<br> This node is:"+str(node["id"])+"<br> it's membership is "+str("unknown")+" It's neighbors are:<br>" + "<br>".join(neighbor_map[node["id"]])
        #
        if node["id"] in node_size.keys():
            #if not labels:
            node["size"] = 1250.0 * node_size[node["id"]]
        node["label"] = str(node["id"])
        node["value"] = len(neighbor_map[node["id"]])
        # st.text(node["id"])
        # st.text(color_code.keys())
        if node["id"] in color_code.keys():
            node["color"] = color_code[node["id"]]
        # else:
        # 	st.text(node["id"])
        # if not labels:
        # node["borderWidth"] = 10
        # node["borderWidthSelected"] = 20
    # nt.show()
    nt.show("test1.html")
    HtmlFile = open("test1.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=750, width=750)
    fig = plt.figure()
    #fig, ax = plt.subplots(figsize=(3, 3))

    for k, v in color_dict.items():
        plt.scatter([], [], c=v, label=k)
    plt.legend(frameon=False,prop={'size':4.0})
    st.pyplot(fig)
    if phys_ == "Yes":
        from PIL import Image
        st.markdown("Some parameter sets can prevent static equilibrium states. For example:")
        #nt.show_buttons(filter_=["physics"])
        st.image(Image.open("rescreen_shot_just_params.png"))


def dont():
    chord = hv.Chord(links)  # .select(value=(5, None))
    # node_color = [color_code[n] for n in H]
    # st.text(links['color'])
    chord.opts(
        opts.Chord(
            cmap="Category20",
            width=250,
            height=250,
            edge_cmap="Category20",
            edge_color=dim("source").str(),
            labels="name",
            node_color=dim("index").str(),
        )
    )


def dontdo():
    """
    sorting_feature = "club"
    hp = hive_plot_n_axes(
                    node_list=nodes,
                    edges=edges,
                    axes_assignments=[
                                    IRG1_indices,
                                    IRG2_indices,
                                    IRG3_indices,
                                    DCMT_ind,
                                    Un_ind,
                    ],
                    sorting_variables=["club", "club", "club", "club", "club"],
                    axes_names=["IRG1", "IRG2", "IRG3", "DCMT", "Unknown"],
                    vmins=[0, -0, 0, -10, -10],
                    vmaxes=[33, 33, 33, 33, 33],
    )

    #fig = hive_plot_viz_mpl(hp)

    ### axes ###

    axis0 = Axis(axis_id="hi_id", start=1, end=5, angle=-30,
                                                     long_name="Mr. Hi Faction\n(Sorted by ID)")
    axis1 = Axis(axis_id="hi_degree", start=1, end=5, angle=30,
                                                     long_name="Mr. Hi Faction\n(Sorted by Degree)")
    axis2 = Axis(axis_id="john_degree", start=1, end=5, angle=180 - 30,
                                                     long_name="John A. Faction\n(Sorted by Degree)")
    axis3 = Axis(axis_id="john_id", start=1, end=5, angle=180 + 30,
                                                     long_name="John A. Faction\n(Sorted by ID)")
    axis4 = Axis(axis_id="irg1_id", start=1, end=5, angle=180,
                                                     long_name="John A. Faction\n(Sorted by ID)")

    axes = [axis0, axis1, axis2, axis3, axis4]

    karate_hp.add_axes(axes)

    ### node assignments ###

    color_dict = {
                    "Unknown": "black",
                    "IRG 3": "green",
                    "IRG 1": "blue",
                    "IRG 2": "yellow",
                    "DCMT": "orange",
    }

    # partition the nodes into "Mr. Hi" nodes and "John A." nodes
    IRG1_nodes = [node.unique_id for node in nodes if node.data['club'] == "IRG 1"]
    IRG2_nodes = [node.unique_id for node in nodes if node.data['club'] == "IRG 2"]
    DCMT_nodes = [node.unique_id for node in nodes if node.data['club'] == "DCMT"]
    hi_nodes = [node.unique_id for node in nodes if node.data['club'] == "IRG 3"]
    john_a_nodes = [node.unique_id for node in nodes if node.data['club'] == "Unknown"]
    #st.text(hi_nodes[0])
    # assign nodes and sorting procedure to position nodes on axis
    karate_hp.place_nodes_on_axis(axis_id="hi_id", unique_ids=hi_nodes,
                                                                                                                      sorting_feature_to_use="loc", vmin=0, vmax=33)
    karate_hp.place_nodes_on_axis(axis_id="hi_degree", unique_ids=hi_nodes,
                                                                                                                      sorting_feature_to_use="degree", vmin=0, vmax=17)
    karate_hp.place_nodes_on_axis(axis_id="john_degree", unique_ids=john_a_nodes,
                                                                                                                      sorting_feature_to_use="degree", vmin=0, vmax=17)
    karate_hp.place_nodes_on_axis(axis_id="john_id", unique_ids=john_a_nodes,
                                                                                                                      sorting_feature_to_use="loc", vmin=0, vmax=33)
    karate_hp.place_nodes_on_axis(axis_id="irg1_id", unique_ids=IRG1_nodes,
                                                                                                                      sorting_feature_to_use="loc", vmin=0, vmax=33)

    ### edges ###

    karate_hp.connect_axes(edges=edges, axis_id_1="hi_degree", axis_id_2="hi_id", c="C0")
    karate_hp.connect_axes(edges=edges, axis_id_1="john_degree", axis_id_2="john_id", c="C1")
    karate_hp.connect_axes(edges=edges, axis_id_1="hi_degree", axis_id_2="john_degree", c="C2")
    karate_hp.connect_axes(edges=edges, axis_id_1="irg1_id", axis_id_2="john_id", c="C3")

    # pull out the location of the John A. and Mr. Hi nodes for visual emphasis later
    john_a_degree_locations = karate_hp.axes["john_degree"].node_placements
    john_a_node = john_a_degree_locations.loc[john_a_degree_locations.loc[:, 'unique_id'] == 33,
                                                                                                                                                                      ['x', 'y']].values.flatten()

    mr_hi_degree_locations = karate_hp.axes["hi_degree"].node_placements
    mr_hi_node = mr_hi_degree_locations.loc[mr_hi_degree_locations.loc[:, 'unique_id'] == 0,
                                                                                                                                                                    ['x', 'y']].values.flatten()

    # plot axes
    fig, ax = axes_viz_mpl(karate_hp,
                                                                                       axes_labels_buffer=1.4)

    # plot nodes
    node_viz_mpl(karate_hp,
                                                     fig=fig, ax=ax, s=80, c="black")

    # highlight Mr. Hi and John. A on the degree axes
    #ax.scatter(john_a_node[0], john_a_node[1],
    #           facecolor="red", edgecolor="black", s=150, lw=2)
    #ax.scatter(mr_hi_node[0], mr_hi_node[1],
    #           facecolor="yellow", edgecolor="black", s=150, lw=2)

    # plot edges
    edge_viz_mpl(hive_plot=karate_hp, fig=fig, ax=ax, alpha=0.7, zorder=-1)

    ax.set_title("Odor 2 Action \nHive Plot", fontsize=20, y=0.9)

    ### legend ###

    # edges

    custom_lines = [Line2D([0], [0], color=f'C{i}', lw=3, linestyle='-') for i in range(3)]


    ax.legend(custom_lines, ["Within Mr. Hi Faction", "Within John A. Faction",
                                                                                                     "Between Factions"],
                                      loc='upper left', bbox_to_anchor=(0.37, 0.35), title="Social Connections")
    st.pyplot(fig)
    """
from scipy.spatial import Delaunay, ConvexHull
from pyveplot import Hiveplot, Axis, Node
import networkx as nx
import random
import base64
import textwrap

def render_svg_small(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width = 900>' % b64
    st.write(html, unsafe_allow_html=True)
#        hub_sort(first,color_code_0,reverse)
def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


def agraph_(first):
    from streamlit_agraph import agraph, Node, Edge, Config

    config = Config(height=500, width=700, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=True,
              collapsible=True)
    #st.text(dir(agraph))
    #agraph(list(first.nodes), (first.edges), config)

def hub_sort(first,color_code_1,reverse):
    c = ['#e41a1c', '#377eb8', '#4daf4a',
         '#984ea3', '#ff7f00', '#ffff33',
         '#a65628', '#f781bf', '#999999',]

    # create hiveplot object
    h = Hiveplot()
    fig = plt.figure()
    # create three axes, spaced at 120 degrees from each other
    h.axes = [Axis(start=20, angle=0,
                   stroke='black', stroke_width=2.1),
              Axis(start=20, angle=120,
                   stroke='black', stroke_width=2.1),
              Axis(start=20, angle=120 + 120,
                   stroke='black', stroke_width=2.1)
              ]

    # create a random Barabasi-Albert network
    g = first

    # sort nodes by degree
    k = list(nx.degree(g))
    k.sort(key=lambda tup: tup[1])

    maxd = np.max([i[1] for i in k])

    # categorize them as high, medium and low degree
    hi_deg = [v[0] for v in k if v[1] > 2*maxd/3]
    md_deg = [v[0] for v in k if v[1] > maxd/3 and v[1] <= 2*maxd/3]
    lo_deg = [v[0] for v in k if v[1] <= maxd/3]

    # place these nodes into our three axes
    for axis, nodes in zip(h.axes,
                           [hi_deg, md_deg, lo_deg]):
        #random.choice(c)
        for v in nodes:
            circle_color = color_code_1[v]
            # create node object
            node = Node(radius=22.5*g.degree(v),
                        label="%s" % (v))
            # add it to axis
            axis.add_node(v, node)
            # once it has x, y coordinates, add a circle
            node.add_circle(fill=circle_color, stroke=circle_color,
                            stroke_width=0.1, fill_opacity=0.65)
            if axis.angle < 180:
                orientation = -1
                scale = 6.5
            else:
                orientation = 1
                scale = 1.5
            # also add a label
            node.add_label("{0}".format(v),
                           angle=axis.angle + 90 * orientation,
                           scale=scale)
            #st.text("node {0}".format(v))

    # iterate through axes, from left to right
    for n in range(-1, len(h.axes) - 1):

        curve_color = 'black'#random.choice(c)
        # draw curves between nodes connected by edges in network
        h.connect_axes(h.axes[n],
                       h.axes[n+1],
                       g.edges,
                       stroke_width=4.5,
                       stroke=curve_color)
    #st.pyplot(fig)

    # save output
    import os
    os.system('rm ba_hiveplot.svg')
    h.save('ba_hiveplot.svg')
    with open('ba_hiveplot.svg',"r") as f:
        lines = f.readlines()
    line_string=''.join(lines)

    render_svg_small(line_string)




def main():

    st.sidebar.title("Odor To Action: Collaboration Survey Data")

    # st.sidebar.markdown("""I talk or directly email with this person (for any reason)...\n""")

    # st.sidebar.markdown("""Graphs loading first plotting spread sheets...\n""")
    try:
        from community import community_louvain

        genre = st.sidebar.radio(
            "Prefered graph layout?",
            (

                "Hive",
                "Physics",
                "Chord",
                "Bundle",
                "List Centrality",
                "Community Mixing",
                "Basic",
                "Lumped Population",
                "Spreadsheet",
                "AdjacencyMatrix",
                "3D"

            ),
        )
    except:
        genre = st.sidebar.radio(
            "Prefered graph layout?",
            (
                "Hive",
                "Chord",
                "Physics",
                "List Centrality",
                "Bundle",
                "Basic",
                "Lumped Population",
                "Spreadsheet",
                "AdjacencyMatrix",
            ),
        )

    my_expander = st.sidebar.beta_expander("Explanation of Threshold")

    my_expander.markdown(
        """
		Problem most people politely answer that they talk to someone a little bit, \
		a bias if which is not corrected \n for hyperconnects everyone to everyone \
		else in a meaningless way. The solution \n to this problem involves thresholding, \
		setting a minimum meaningful level of \n communication collaboration, \
		The higher the threshold the more you \n reduce connections"""
    )
    my_expander = st.beta_expander("Set threshold")
    #if genre == "Bundle":
    #    threshold = my_expander.slider("Select a threshold value", 0.0, 10.0, 5.0, 1.0)
    #else:
    threshold = my_expander.slider("Select a threshold value", 0.0, 8.0, 5.0, 1.0)
    # st.write("Values:", threshold)
    (
        df2,
        names,
        ratercodes,
        legend,
        color_code,
        color_dict,
        color_code_0,
        sheet,
        popg,
        hc
    ) = get_frame(threshold)

    #fig = plt.figure()
    #for k, v in color_dict.items():
    #    plt.scatter([], [], c=v, label=k)
    #plt.legend(frameon=False,prop={'size':24})
    #fig.tight_layout()
    #plt.axis("off")
    #my_expander = st.sidebar.beta_expander("Color coding of most plots")
    #my_expander.markdown(
    #    """ Excepting for chord and hive, which are time consuming to code"""
    #)
    #my_expander.pyplot(fig)
    inboth = set(names) & set(ratercodes)
    notinboth = set(names) - set(ratercodes)
    allcodes = set(names) or set(ratercodes)
    first = nx.DiGraph()

    for i, row in enumerate(allcodes):
        if i != 0:
            if row[0] != 1 and row[0] != 0:
                first.add_node(row[0], name=row)  # ,size=20)
    adj_mat_dicts = []
    conns = {}
    cc = copy.copy(color_code_0)
    for i, idx in enumerate(df2.index):
        for j, col in enumerate(df2.columns):
            if col not in cc.keys():
                cc[col] = hc[col]
            if idx not in color_code_0.keys():
                cc[col] = hc[col]

    for i, idx in enumerate(df2.index):
        for j, col in enumerate(df2.columns):
            weight = float(df2.iloc[i, j])
            if idx != col:
                if float(weight) > threshold:
                    adj_mat_dicts.append({"src": idx, "tgt": col, "weight": weight})
                    first.add_edge(idx, col, weight=weight)

            if not popg.has_edge(cc[idx], cc[col]):
                popg.add_edge(cc[idx], cc[col], weight=weight)
            else:
                e = popg.get_edge_data(cc[idx], cc[col])
                weight = weight + e["weight"]
                popg.add_edge(cc[idx], cc[col], weight=weight)


    first.remove_nodes_from(list(nx.isolates(first)))
    adj_mat = pd.DataFrame(adj_mat_dicts)
    try:
        encoded = {v: k for k, v in enumerate(first.nodes())}
    except:
        encoded = {v: k for k, v in enumerate(adj_mat.columns)}
    adj_mat = adj_mat[adj_mat["weight"] != 0]

    link = dict(
        source=[encoded[i] for i in list(adj_mat["src"].values)],
        target=[encoded[i] for i in list(adj_mat["tgt"].values)],
        value=[i * 3 for i in list(adj_mat["weight"].values)],
    )
    adj_mat2 = pd.DataFrame(link)
    adj_mat3 = adj_mat[adj_mat["weight"] != 0]


    encoded = {v: k for k, v in enumerate(first.nodes())}
    reverse = {v: k for k, v in encoded.items()}
    G = nx.relabel_nodes(first, encoded, copy=True)
    edges = np.array(G.edges)

    node_ids, degrees = np.unique(edges, return_counts=True)

    for i, (node_id, degree) in enumerate(zip(node_ids, degrees)):
        if not reverse[node_id] in color_code_0.keys():
            color_code_0[reverse[node_id]] = hc[reverse[node_id]]
            reverse[node_id] = hc[reverse[node_id]]

    #unknownids = [k for k,v in cc.items() if v=="Unknown"]
    #st.text(unknownids)
    if genre == "List Centrality":
        hub_sort(first,color_code,reverse)

        list_centrality(first)
    if genre == "Spreadsheet":
        st.markdown("Processed anonymized network data that is visualized")
        st.markdown(get_table_download_link_csv(df2), unsafe_allow_html=True)
        st.markdown("Anonymized raw survey data")
        st.markdown(get_table_download_link_csv(sheet), unsafe_allow_html=True)
        #st.beta_expander()
        my_expander = st.beta_expander("Numeric mapping of survery question answers")
        my_expander.write(legend)
        my_expander = st.beta_expander("Collapsed/Expand Numeric Spread sheet")
        my_expander.table(df2)
        my_expander = st.beta_expander("Collapsed/Expand Raw Spread sheet")
        my_expander.table(sheet)

    #try:
    if genre == "Community Mixing":
        my_expander = st.beta_expander("Explanation of Community Partitions")
        my_expander.markdown("""Communities in the graph on the left are not IRG 1-3, but instead communities found by blind network analysis. It's appropritate to use a different color code for the five inferred communities. \
        For contrast in the graph on the right, machine driven community detection clusters persist, but now nodes are color coded IRG-1-3 \n \
        This suggests that the formal memberships eg. \"IRG 1\" does not determine the machine generated communities. In otherwords spontaneuosly emerging community groups may be significantly different to formal group assignments.
        The stochastic community detection algorithm uses a differently seeded random number generator every time so the graph appears differently each time the function is called.
        The algorithm is called Louvain community detection. The Louvain Community algorithm detects 5 communities, but only 2 communities with membership >=3. A grey filled convex hull is drawn around each of the two larger communities.
        """)

        community(first,color_code,color_dict)
    #except:
    #    pass
    if genre == "3D":
        st.markdown("in development")
        g = first

        links = copy.copy(adj_mat)
        links.rename(
            columns={"weight": "value", "src": "source", "tgt": "target"}, inplace=True
        )
        links = links[links["value"] != 0]
        Edges=[(encoded[src],encoded[tgt]) for src,tgt in zip(links['source'], links['target'])]
        G=ig.Graph(Edges, directed=False)

        layt=G.layout('kk', dim=3) # plot network with the Kamada-Kawai layout algorithm

        labels=[]
        group=[]

        for node in links['source']:
           labels.append(str(node))
           group.append(color_code[node])
           #st.text(node)
           #st.text(color_code[node])

        Xn=[]
        Yn=[]
        Zn=[]
        N=len(g.nodes)
        for k in range(N):
          Xn+=[layt[k][0]]
          Yn+=[layt[k][1]]
          Zn+=[layt[k][2]]

        Xe=[]
        Ye=[]
        Ze=[]

        for e in Edges:
          Xe+=[layt[e[0]][0],layt[e[1]][0],None]# x-coordinates of edge ends
          Ye+=[layt[e[0]][1],layt[e[1]][1],None]
          Ze+=[layt[e[0]][2],layt[e[1]][2],None]

        trace1=go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=dict(color='rgb(125,125,125)', width=1),hoverinfo='none')

        trace2=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', name='Researchers',
                           marker=dict(symbol='circle',color=group, size=6,colorscale='Viridis',
                              line=dict(color='rgb(50,50,50)', width=0.5)))#,text=labels,hoverinfo='text'))

        axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

        layout = go.Layout(
                 title="(3D visualization) Can be rotated",
                 width=1000,
                 height=1000,
                 showlegend=False,
                 scene=dict(
                     xaxis=dict(axis),
                     yaxis=dict(axis),
                     zaxis=dict(axis),
                ))

        data=[trace1, trace2]

        fig=go.Figure(data=data, layout=layout)
        st.write(fig)
    if genre == "Physics":
        physics(first, adj_mat_dicts, color_code,color_code_0,color_dict)

    if genre == "Lumped Population":
        population(cc, popg, color_dict)
    #if genre == "hive-degree":
        #colored_hive_axis(first,color_code_0,reverse)
        #agraph_(first)
    if genre == "Hive":
        from hiveplotlib import Axis, Node, HivePlot

        # convert `networkx` edges and nodes into `hiveplotlib`-ready structures
        G = first
        encoded = {v: k for k, v in enumerate(first.nodes())}
        reverse = {v: k for k, v in encoded.items()}

        G = nx.relabel_nodes(G, encoded, copy=True)
        edges = np.array(G.edges)

        # pull out degree information from nodes for later use
        node_ids, degrees = np.unique(edges, return_counts=True)

        #nodes = np.array(G.nodes)
        nodes = []

        IRG1_indices = []
        IRG2_indices = []
        IRG3_indices = []
        DCMT_ind = []  # ,Un_ind
        #st.text(len(color_code_0))
        for i, (node_id, degree) in enumerate(zip(node_ids, degrees)):
            if not reverse[node_id] in color_code_0.keys():
                color_code_0[reverse[node_id]] = hc[reverse[node_id]]
                reverse[node_id] = hc[reverse[node_id]]

            temp_node = Node(unique_id=node_id, data=G.nodes.data()[node_id])
            nodes.append(temp_node)
        for i, (node_id, degree) in enumerate(zip(node_ids, degrees)):
            # store the index number as a way to align the nodes on axes
            G.nodes.data()[node_id]["loc"] = node_id
            # also store the degree of each node as another way to align nodes on axes
            G.nodes.data()[node_id]["degree"] = degree
            # G.nodes.data()[node_id]['club'] =

            if reverse[node_id] in color_code_0.keys():
                # = color_code_0[reverse[node_id]]
                if color_code_0[reverse[node_id]] == "IRG 1":
                    IRG1_indices.append(i)
                    G.nodes.data()[node_id]["IRG 1"] = 1
                    G.nodes.data()[node_id]["club"] = 1
                if color_code_0[reverse[node_id]] == "IRG 2":
                    IRG2_indices.append(i)
                    G.nodes.data()[node_id]["IRG 2"] = 2
                    G.nodes.data()[node_id]["club"] = 2

                if color_code_0[reverse[node_id]] == "IRG 3":
                    IRG3_indices.append(i)
                    G.nodes.data()[node_id]["IRG 3"] = 3
                    G.nodes.data()[node_id]["club"] = 3

                # st.text(IRG3_indices)
                if color_code_0[reverse[node_id]] == "DCMT":
                    DCMT_ind.append(i)
                    G.nodes.data()[node_id]["DCMT"] = 4
                    G.nodes.data()[node_id]["club"] = 4

        temp = list(set(color_code_0.values()))
        hp = hive_plot_n_axes(
            node_list=nodes,
            edges=edges,
            axes_assignments=[
                IRG1_indices,
                IRG2_indices,
                IRG3_indices,
                DCMT_ind,
            ],
            sorting_variables=["club", "club", "club", "club"],
            axes_names=temp,
            vmins=[0, 0, 0, 0],
            vmaxes=[2, 2, 2, 2],
            orient_angle=30,
        )


        # change the line kwargs for edges in plot
        hp.add_edge_kwargs(
            axis_id_1=temp[0], axis_id_2=temp[1], c=f"C0", lw=1.5, alpha=0.5, zorder=10
        )
        hp.add_edge_kwargs(
            axis_id_1=temp[1], axis_id_2=temp[2], c=f"C2", lw=1.5, alpha=0.5, zorder=10
        )
        #hp.add_edge_kwargs(
        ##    axis_id_1=temp[0], axis_id_3=temp[2], c=f"C1", lw=1.5, alpha=0.5, zorder=10
        #)

        # st.text(temp[2])
        hp.place_nodes_on_axis(
            axis_id=temp[0],
            unique_ids=[nodes[i].data["loc"] for i in IRG1_indices],
            sorting_feature_to_use="loc",
            vmin=0,
            vmax=33,
        )
        hp.place_nodes_on_axis(
            axis_id=temp[1],
            unique_ids=[nodes[i].data["loc"] for i in IRG2_indices],
            sorting_feature_to_use="loc",
            vmin=0,
            vmax=33,
        )
        hp.place_nodes_on_axis(
            axis_id=temp[2],
            unique_ids=[nodes[i].data["loc"] for i in IRG3_indices],
            sorting_feature_to_use="loc",
            vmin=0,
            vmax=33,
        )
        hp.place_nodes_on_axis(
            axis_id=temp[3],
            unique_ids=[nodes[i].data["loc"] for i in DCMT_ind],
            sorting_feature_to_use="loc",
            vmin=0,
            vmax=33,
        )
        #hp.place_nodes_on_axis(
        #    axis_id=temp[3],
        #    unique_ids=[nodes[i].data["loc"] for i in Un_ind],
        #    sorting_feature_to_use="loc",
        #    vmin=0,
        #    vmax=33,
        #)

        hp.connect_axes(edges=edges, axis_id_1=temp[0], axis_id_2=temp[1], c="C1")
        hp.connect_axes(edges=edges, axis_id_1=temp[1], axis_id_2=temp[2], c="C2")
        hp.connect_axes(edges=edges, axis_id_1=temp[0], axis_id_2=temp[2], c="C2")
        hp.connect_axes(edges=edges, axis_id_1=temp[2], axis_id_2=temp[3], c="C3")
        hp.connect_axes(edges=edges, axis_id_1=temp[3], axis_id_2=temp[1], c="C1")
        hp.connect_axes(edges=edges, axis_id_1=temp[3], axis_id_2=temp[0], c="C0")
        #hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[0], c="C7")
        #hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[1], c="C8")
        #hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[2], c="C9")
        #hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[3], c="C10")

        fig, ax = hive_plot_viz_mpl(hive_plot=hp)


        # john_a_degree_locations = \
        # karate_hp.axes["john_degree"].node_placements
        # [nodes[i]
        # for i in IRG3_indices:
        #    st.text(nodes[i].data['loc'])
        # ax.scatter(x, y,
        #           facecolor="red", edgecolor="black", s=150, lw=2)
        # ax.scatter(mr_hi_node[0], mr_hi_node[1],
        #           facecolor="yellow", edgecolor="black", s=150, lw=2)
        # my_expander = st.side_bar.beta_expander("Explanation of Hive")

        my_expander = st.beta_expander("Explanation of Hive")

        my_expander.markdown(
            """This predominantly shows between group connectivity. The total node number in each category is reduced compared to other graphs, as only nodes
			are shown which can project externally from their respective groups.
			"""
        )

        st.pyplot(fig)
        my_expander = st.beta_expander("Explanation of Second Hive")

        my_expander.markdown(
            """This graphically shows network centrality from densely into connected (hub) to sparsely interconnected.
			"""
        )

        hub_sort(first,color_code,reverse)


    if genre == "Bundle":
        my_expander = st.beta_expander("show labels?")
        labels_ = my_expander.radio("Would you like to label nodes?", ("No", "Yes"))
        if labels_ == "Yes":
            labels = True
        if labels_ == "No":
            labels = False
        my_expander = st.beta_expander("Bundling explanation")

        my_expander.markdown(
            """The graph type below is called edge bundling. "Bundling" connecting cables simplifies the visualization.\n \
			 Think of it like internet cables which are bundled. Internet backbones connect places far \n \
			 apart as to economize wiring material. Conservation of wire material is also seen in the nervous system.
             In the corpus callosum and spinal column convergent paths are constricted into relatively narrower bundles.""")

        fig4 = data_shade(first, color_code, adj_mat, color_dict, labels)
        st.pyplot(fig4)
    #import streamlit as st
    if genre== "cyto":
        #from ipycytoscape import CytoscapeWidget
        #cyto = CytoscapeWidget()
        #cyto.graph.add_graph_from_networkx(first)
        #st.text(dir(cyto))
        #components.html(raw_html)
        #cyto_data = nx.cytoscape_data(first)
        #cyto_graph = nx.cytoscape_graph(cyto_data)
        #st.text(type(cyto_graph))
        #st.text(cyto_data["elements"])
        #st.text(cyto_data.keys())
        #st.text(cyto_data['directed'])
        #st.text(cyto.cytoscape_style)
        #st.text(cyto.cytoscape_layout)
        #st.text(color_dict)
        #st.text(color_code)

        G = first
        pos=nx.fruchterman_reingold_layout(G)
        A = nx.to_pandas_adjacency(first)
        nodes = [
            {
                'data': {'id': node, 'label': node},
                'position': {'x': 500*pos[node][0], 'y': 500*pos[node][1]},
                'color':color_code[node]
                #'locked': 'true'
            }
            for node in G.nodes if node in color_code
        ]

        edges = []
        for col in A:
            for row, value in A[col].iteritems():
                if {'data': {'source': row, 'target': col}} not in edges and row != col:
                    edges.append({'data': {'source': col, 'target': row}})

        for edge in edges:
            edge['data']['weight'] = 0.1*A.loc[edge['data']['source'], edge['data']['target']]

        elements1 = nodes + edges


        #from streamlit_cytoscapejs import st_cytoscapejs

        #elements = [{"data":cyto_data['elements']}]
        #st.text(elements)

        import streamlit_bd_cytoscapejs
        elements = elements1#cyto_data#[{"data":cyto_data['elements']}]
        #st.text(cyto_data['elements'])
        layout = {'name': 'random'}
        layout = {'name': 'preset'}
        #grid
        #circle
        #concentric
        #breadthfirst
        #cose
        stylesheet=[{
            'selector': 'node',
            'style': {
                'label': 'data(id)'
            }
        },
        {
            'selector': 'edge',
            'style': {
                # The default curve style does not work with certain arrows
                'curve-style': 'bezier'
            }
        },
        {
            'selector': '#BA',
            'style': {
                'source-arrow-color': 'red',
                'source-arrow-shape': 'triangle',
                'line-color': 'red'
            }
        },
        {
            'selector': '#DA',
            'style': {
                'target-arrow-color': 'blue',
                'target-arrow-shape': 'vee',
                'line-color': 'blue'
            }
        },
        {
            'selector': '#BC',
            'style': {
                'mid-source-arrow-color': 'green',
                'mid-source-arrow-shape': 'diamond',
                'mid-source-arrow-fill': 'hollow',
                'line-color': 'green',
            }
        },
        {
            'selector': '#CD',
            'style': {
                'mid-target-arrow-color': 'black',
                'mid-target-arrow-shape': 'circle',
                'arrow-scale': 2,
                'line-color': 'black',
            }
        }
        ]

        node_id = streamlit_bd_cytoscapejs.st_bd_cytoscape(
            elements,
            layout=layout,
            key='foo'
        )
        st.write(node_id)

        import dash_cytoscape as cyto
        import dash_html_components as html

        #app = dash.Dash(__name__)
        #layout = html.Div([
        cyto.Cytoscape(
            id='cytoscape',
            elements=elements,
            layout={'name': 'preset'}
            )
        #])
        #st.text(dir(cyto))
        #st.write(cyto)
        #st.text()
        #st.write(layout.to_plotly_json())
        #components.html(layout.to_plotly_json())

    if genre == "Basic":
        #'plt.rcParams['legend.title_fontsize'] = 'xx-large'

        my_expander = st.beta_expander("show labels?")

        labels_ = my_expander.radio("Would you like to label nodes?", ("No", "Yes"))
        if labels_ == "Yes":
            labels_ = True
        if labels_ == "No":
            labels_ = False
        exp = st.beta_expander("Information about Force Directed Layout")
        exp.markdown(
            """This is probably not the most informative layout option. Contrast this force directed layout network layout with bundling (wire cost is not economized here).
            The basic force directed layout is very similar to the physics engine layout, but without interactivity.
            Qoute from wikipedia:
            'Force-directed graph drawing algorithms assign forces among the set of edges and the set of nodes of a graph drawing. Typically, spring-like attractive forces based on Hooke's law are used to attract pairs of endpoints of the graph's edges towards each other, while simultaneously repulsive forces like those of electrically charged particles based on Coulomb's law are used to separate all pairs of nodes. In equilibrium states for this system of forces, the edges tend to have uniform length (because of the spring forces), and nodes that are not connected by an edge tend to be drawn further apart (because of the electrical repulsion). Edge attraction and vertex repulsion forces may be defined using functions that are not based on the physical behavior of springs and particles; for instance, some force-directed systems use springs whose attractive force is logarithmic rather than linear.'
            \n
            https://en.wikipedia.org/wiki/Force-directed_graph_drawing \n
            What this means is conflicting forces of attraction, and repulsion determine node position.
            node centrality does not necessarily determine a central position.
            Also nodes can be central because of high in-degree out-degree or both.
            """
        )
        H = first.to_undirected()


        centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
        edge_thickness = [v * 20000 for v in centrality.values()]
        node_size = [v * 20000 for v in centrality.values()]

        # compute community structure
        lpc = nx.community.label_propagation_communities(H)
        community_index = {n: i for i, com in enumerate(lpc) for n in com}

        #### draw graph ####
        fig, ax = plt.subplots(figsize=(20, 15))
        # fig, ax = plt.subplots(figsize=(15,15))

        pos = nx.spring_layout(H, k=0.05, seed=4572321, scale=1)

        node_color = [color_code[n] for n in H]
        srcs = list(adj_mat["src"].values)

        srcs = []
        for e in H.edges:
            src = color_code[e[0]]
            srcs.append(src)

        # for ind,seg in enumerate(segments):
        # 	 ax.plot(seg[:,0], seg[:,1],c=color_code[srcs[ind]],alpha=0.35,linewidth=0.25*widths[ind])

        nx.draw_networkx_nodes(
            H,
            pos=pos,
            node_color=node_color,
            node_size=node_size,
            alpha=0.5,
            linewidths=2,
        )

        labels = {}
        for node in H.nodes():
            # set the node name as the key and the label as its value
            labels[node] = node
        if labels_:
            nx.draw_networkx_labels(H, pos, labels, font_size=16, font_color="r")

        axx = fig.gca()  # to get the current axis
        axx.collections[0].set_edgecolor("#FF0000")
        nx.draw_networkx_edges(
            H, pos=pos, edge_color=srcs, alpha=0.5, width=list(adj_mat["weight"].values)
        )

        # Title/legend
        font = {"color": "k", "fontweight": "bold", "fontsize": 20}
        ax.set_title("network", font)
        # Change font color for legend
        font["color"] = "b"

        ax.text(
            0.80,
            0.06,
            "node size = betweeness centrality",
            horizontalalignment="center",
            transform=ax.transAxes,
            fontdict=font,
        )

        # Resize figure for label readibility
        ax.margins(0.1, 0.05)
        fig.tight_layout()
        plt.axis("off")

        for k, v in color_dict.items():
            plt.scatter([], [], c=v, label=k)

        plt.legend(frameon=False,prop={'size':24})
        #leg = ax.legend()
        #leg.set_title()
        st.pyplot(fig)

    adj_mat = pd.DataFrame(adj_mat_dicts)
    narr = nx.to_pandas_adjacency(first)

    ideo_colors = [
        "rgba(244, 109, 67, 0.75)",
        "rgba(253, 174, 97, 0.75)",
        "rgba(254, 224, 139, 0.75)",
        "rgba(217, 239, 139, 0.75)",
        "rgba(166, 217, 106, 0.75)",
        "rgba(244, 109, 67, 0.75)",
        "rgba(253, 174, 97, 0.75)",
        "rgba(254, 224, 139, 0.75)",
        "rgba(217, 239, 139, 0.75)",
        "rgba(166, 217, 106, 0.75)",
    ]

    if genre == "AdjacencyMatrix":

        my_expander = st.beta_expander("Explanation of Adjacency Clustergrams")

        my_expander.markdown(
            """Clustergrams are re-sorted adjacency matrices, thats why their diaganols do not appear to be zero. \n
			The sorting maximizes cluster size.

            In all three graphs below, only every second node is labelled on x,y axis. This is so as not over crowd the
            axis labels. If you look closely at the pixels, pixels vary at double the frequency of the node labels.
			"""
        )

        g = sns.clustermap(df2)
        st.pyplot(g)

        st.markdown(
            "un sorted interactive adjacency matrix (data not organized to emphasise clusters)"
        )
        st.markdown(
            "Diagnols are still not zero, because column names and row names are not sorted."
        )

        fig = plot_imshow_plotly(df2)
        st.write(fig)
        st.markdown("sorted interactive clustergram of adjacency matrix")

        columns = list(df2.columns.values)
        rows = list(df2.index)
        try:
            figure = dashbio.Clustergram(
                data=df2.loc[rows].values,
                column_labels=columns,
                row_labels=rows,
                color_threshold={"row": 0, "col": 70},
                hidden_labels="row",
                height=800,
                width=800,
            )

            st.write(figure)
        except:
            pass
    if genre == "Chord":
        # https://docs.bokeh.org/en/0.12.3/docs/gallery/chord_chart.html
        #from bokeh import Chord
        #nodes = data['nodes']
        #links = data['links']

        #nodes_df = pd.DataFrame(nodes)
        #links_df = pd.DataFrame(links)

        #source_data = links_df.merge(nodes_df, how='left', left_on='source', right_index=True)
        #source_data = source_data.merge(nodes_df, how='left', left_on='target', right_index=True)
        #source_data = source_data[source_data["value"] > 5]

        #chord_from_df = Chord(source_data, source="name_x", target="name_y", value="value")
        #st.markdown(""" clicking on a node highlights its direct projections""")

        H = first.to_undirected()
        st.markdown("Betweeness Centrality:")
        st.markdown("Top to bottom node id from most central to least:")

        centrality = nx.betweenness_centrality(H, endpoints=True)
        df = pd.DataFrame([centrality])
        df = df.T
        df.sort_values(0, axis=0, ascending=False, inplace=True)
        bc = df
        bc.rename(columns={0:'centrality value'},inplace=True)
        st.write(bc.head())
        #st.markdown("In degree Centrality:")
        #st.markdown("Top to bottom node id from most central to least:")

        temp = pd.DataFrame(first.nodes)
        nodes = hv.Dataset(temp[0])

        links = copy.copy(adj_mat)
        links.rename(
            columns={"weight": "value", "src": "source", "tgt": "target"}, inplace=True
        )
        links = links[links["value"] != 0]

        Nodes_ = set(
            links["source"].unique().tolist() + links["target"].unique().tolist()
        )
        Nodes = {node: i for i, node in enumerate(Nodes_)}

        df_links = links.replace({"source": Nodes, "target": Nodes})
        for k in Nodes.keys():
            if k not in color_code_0.keys():
                color_code_0[k] = "Unknown"

        df_nodes = pd.DataFrame(
            {
                "index": [idx for idx in Nodes.values()],
                "name": [name for name in Nodes.keys()],
                "colors": [color_code_0[k] for k in Nodes.keys()],
            }
        )
        dic_to_sort = {}
        for i,kk in enumerate(df_nodes["name"]):
            dic_to_sort[i] = color_code_0[k]

        t = pd.Series(dic_to_sort)
        df_nodes['sort']=t#pd.Series(df_links.source)
        df_nodes.sort_values(by=['sort'],inplace=True)

        dic_to_sort = {}
        for i,kk in enumerate(df_links["source"]):
            k = df_nodes.loc[kk, "name"]
            # st.text(k)
            if k not in color_code_0.keys():
                color_code_0[k] = "Unknown"
            df_nodes.loc[kk,"colors"] = color_code_0[k]
            dic_to_sort[i] = color_code_0[k]

        pd.set_option("display.max_columns", 11)
        hv.extension("bokeh")
        hv.output(size=200)
        t = pd.Series(dic_to_sort)
        df_links['sort']=t#pd.Series(df_links.source)
        df_links.sort_values(by=['sort'],inplace=True)
        #df_links['colors'] = None
        categories = np.unique(df_links["sort"])
        colors = np.linspace(0, 1, len(categories))
        colordicth = dict(zip(categories, colors))

        df_links["Color"] = df_links["sort"].apply(lambda x: float(colordicth[x]))
        #for i,row in df_links.iterrows():
        #    st.text(i)
        #    if row[-1]['sort'] == "IRG 1":
        #        row[-1]
            #if row[-2] == "IRG 1":
                #if df_links.loc[i,'sort'] == "IRG 1":
                #st.text(df_links.loc[i,'sort'])

            #df_links.loc[i,'colors']
        #df_nodes["index"] = df_links["Color"]
        #st.write(df_links)
        #st.write()

        # https://geomdata.gitlab.io/hiveplotlib/karate_club.html
        # Todo make hiveplot
        #
        #st.text(chord.transform)
        #colors,y = chord.transform(chord,"Color")
        #st.text(colors)
        #st.text(dir(chord))
        #from bokeh.sampledata.les_mis import data

        #links = pd.DataFrame(data['links'])
        #nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
        #hv.Chord((links, nodes)).select(value=(5, None)).opts(

        #st.write(hv.render((chordt), backend="bokeh"))

        #st.text(links.head())
        #st.text(links.tail())
        #from chord3 import doCircleRibbonGraph
        #labels = first.nodes
        colors = df_links["Color"].values
        #temp = nx.to_pandas_adjacency(first)
        #temp = temp[temp!=0]
        #temp = temp[temp!=np.nan]
        #st.write(temp)
        #doCircleRibbonGraph(temp, labels, colors, plot_size=400, title="Phd Country")
        #(matrix, labels, colors, plot_size=400, title="Phd Country")
        nodes = hv.Dataset(df_nodes, "index")
        st.write(nodes)
        st.write(df_links)
        df_links["index"] = df_links["Color"]
        chord = hv.Chord((df_links, nodes))#.opts.Chord(cmap='Category20', edge_color=dim('source').astype(str), node_color=dim('index').astype(str))
          # .select(value=(5, None))


        chord.opts(
            opts.Chord(
                cmap="Category20",
                edge_cmap="Category20",
                edge_color=dim("sort").str(),
                width=350,
                height=350,
                labels="Color"
            )
        )
        # st.markdown("Chord layout democratic")
        hv.save(chord, "chord2.html", backend="bokeh")
        HtmlFile2 = open("chord2.html", "r", encoding="utf-8")
        source_code2 = HtmlFile2.read()
        components.html(source_code2, height=750, width=750)

    def dontdo():

        edges_df = links.reset_index(drop=True)
        graph = hv.Graph(edges_df)
        # opts.defaults(opts.Nodes(size=5, padding=0.1))
        from holoviews.operation.datashader import (
            datashade,
            dynspread,
            directly_connect_edges,
            bundle_graph,
            stack,
        )

        # st.markdown("bundling + chord")
        # st.markdown("Able to show that not everything is connected to everything else")
        """
		circular = bundle_graph(graph)
		datashade(circular, width=500, height=500) * circular.nodes
		st.write(hv.render((circular), backend="bokeh"))
		st.markdown("clustergram of adjacency matrix: These don't look the same as sorting algorithms are different")
		"""

    # flux = np.array([[11975,  5871, 8916, 2868],
    #  [ 1951, 10048, 2060, 6171],
    #  [ 8010, 16145, 8090, 8045],
    #  [ 1013,   990,  940, 6907]
    # ])
    # flux = narr

    # ax = plt.axes([0,0,1,1])
    # from chord4 import chordDiagram
    # nodePos = chordDiagram(flux, ax, colors=[hex2rgb(x) for x in ['#666666', '#66ff66', '#ff6666', '#6666ff']])
    # nodePos = chordDiagram(flux, ax)
    # ax.axis('off')
    # prop = dict(fontsize=16*0.8, ha='center', va='center')
    # nodes = ['non-crystal', 'FCC', 'HCP', 'BCC']
    # for i in range(4):
    #    ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)

    # plt.savefig("example.png", dpi=600,
    #        transparent=True,
    #        bbox_inches='tight', pad_inches=0.02)
    # st.pyplot(fig)
    # circular = bundle_graph(graph)
    # chord = hv.Chord(links)
    # datashade(chord, width=300, height=300) #* circular.nodes
    # overlay.opts(opts.Graph(edge_line_color='white', edge_hover_line_color='blue', padding=0.1))
    # st.write(hv.render((chord), backend="bokeh"))

    # st.write(chord)
    # st.write(links)
    # st.write(nodes.data.head())
    # chord = hv.Chord((links,nodes))#.select(value=(5, None))
    # chord.opts(opts.Chord(cmap='Category20',
    # 		   edge_cmap='Category20', edge_color=dim('source').str(),
    #           labels='name', node_color=node_color))
    # st.write(hv.render((chord), backend="bokeh"))
    # st.hvplot(chord)
    # st.write(hv.render(chord, backend="bokeh"))

    # st.write(adj_mat3)
    # chord3 = chord2.make_filled_chord(adj_mat3)
    # st.write(chord3)

    # nodes = first.nodes
    # edges = first.edges
    # value_dim = [i*10 for i in adj_mat["weight"]]
    # careers = hv.Sankey((edges, nodes), ['From', 'To'])#, vdims=value_dim)

    # careers.opts(
    #    opts.Sankey(labels='label', label_position='right', width=900, height=300, cmap='Set1',
    #                edge_color=dim('To').str(), node_color=dim('index').str()))
    # careers.write(hv.render(graph, backend="bokeh"))

    # for trace in edge_trace:
    # 	fig.add_trace(trace)  # Add node trace
    # fig.add_trace(node_trace)  # Remove legend

    # fig.show()


if __name__ == "__main__":

    main()


def dontdo():

    link = dict(source=adj_mat["src"], target=adj_mat["tgt"], value=adj_mat["weight"])

    # generate_sankey_figure(list(first.nodes), adj_mat,title = 'Sankey Diagram')

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(first.nodes()),  # ["A1", "A2", "B1", "B2", "C1", "C2"],
                    color="blue",
                ),
                link=dict(
                    source=adj_mat["src"],
                    target=adj_mat["tgt"],
                    value=[i * 10 for i in adj_mat["weight"]],
                ),
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)

    st.write(fig)
    link = dict(
        source=adj_mat["src"],
        target=adj_mat["tgt"],
        value=[i * 10 for i in adj_mat["weight"]],
    )

    data = go.Sankey(link=link)

    # fig3 = go.Figure(data)
    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",  # transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # transparent 2nd background
        xaxis={"showgrid": False, "zeroline": False},  # no gridlines
        yaxis={"showgrid": False, "zeroline": False},  # no gridlines
    )  # Create figure
    layout["width"] = 925
    layout["height"] = 925

    fig3 = go.Figure(data, layout=layout)  # Add all edge traces
    st.write(fig3)
