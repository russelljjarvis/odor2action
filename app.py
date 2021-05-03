"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)
"""
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
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import shelve
import streamlit as st
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


def data_shade(graph, color_code, adj_mat, color_dict, labels_=False):

    nodes = graph.nodes
    # orig_pos=nx.get_node_attributes(graph,'pos')

    nodes_ind = [i for i in range(0, len(graph.nodes()))]
    redo = {k: v for k, v in zip(graph.nodes, nodes_ind)}
    # pos = nx.spring_layout(H, k=0.05, seed=4572321, scale=1)

    pos_ = nx.spring_layout(graph, scale=1, k=0.05, seed=4572321)
    # node_color = [community_index[n] for n in graph]
    H = graph.to_undirected()
    centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
    node_size = [v * 50000 for v in centrality.values()]

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
    st.markdown(
        'The graph type below is called edge bundling, it gets rid of "hair ball effect"'
    )
    st.markdown(
        'Think of it like internet cables "bundled" backbones connect places far apart as to economize wiring material.'
    )
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
            linewidth=0.25 * widths[ind],
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
    plt.legend()

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
            xlsx_file0 = Path("o2anetmap2021.xlsx")
            xlsx_file1 = Path("o2anetmap.xlsx")
            wb_obj0 = openpyxl.load_workbook(xlsx_file0)
            wb_obj1 = openpyxl.load_workbook(xlsx_file1)

            # Read the active sheet:
            worksheet0 = wb_obj0.active
            worksheet1 = wb_obj1.active

            df3 = pd.DataFrame(worksheet0.values)
            df2 = pd.DataFrame(worksheet1.values)
            import copy

            df2 = pd.concat([df3, df2])
            sheet = copy.copy(df2)
            color_code_0 = {
                k: v for k, v in zip(df2[0], df2[1]) if k not in "Rater Code"
            }

            color_dict = {
                "Unknown": "black",
                "IRG 3": "green",
                "IRG 1": "blue",
                "IRG 2": "yellow",
                "DCMT": "orange",
            }
            color_code_1 = {}

            # if row[0] != 1 and row[0] != 0:
            popg = nx.DiGraph()
            # ,size=20)

            for k, v in color_code_0.items():

                if v not in popg.nodes:
                    # st.text(v)
                    popg.add_node(v, name=v)
                color_code_1[k] = color_dict[v]
            # st.text(color_code_1)
            col_to_rename = df2.columns
            # st.write(df2)
            ratercodes = df2[0][1::]
            row_names = list(df2.T[0].values)
            # st.text(row_names)
            # st.text(list(df2.T[0].values)[-1])
            row_names.append(list(df2.T[0].values)[-1])
            row_names = row_names[2::]
            # st.text(row_names)
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

            row_names = range(0, len(df2.columns) + 1, 1)
            to_rename = {k: v for k, v in zip(row_names, names)}

            r_names = df2.index.values
            to_rename_ind = {v: k for k, v in zip(df2[0], r_names)}
            del df2[0]
            del df2[1]
            # del df2[112]
            # del df2[113]
            df2.drop(0, inplace=True)
            df2.drop(1, inplace=True)
            # try:

            # except:
            # 	pass
            df2.rename(columns=to_rename, inplace=True)
            df2.rename(index=to_rename_ind, inplace=True)

            # uniq_col = {k: k for k in list(set(df2.columns))}
            # comm = False
            # if comm:
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
            df3 = df2[df2.columns[0 : int(len(df2.columns) / 2)]]
            df2 = df2[df2.columns[int(len(df2.columns) / 2) + 1 : -1]]
            import copy

            df4 = copy.copy(df2)
            for col in df2.columns:
                if col in df3.columns and col in df2.columns:
                    df4[col] = df2[col] + df3[col]
                else:
                    df4[col] = df2[col]
            df2 = df4
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


def df_to_plotly(df, log=False):
    return {"z": df.values.tolist(), "x": df.columns.tolist(), "y": df.index.tolist()}


def plot_df_plotly(sleep_df):
    fig = go.Figure(data=go.Heatmap(df_to_plotly(sleep_df, log=True)))
    st.write(fig)


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


def main():

    st.sidebar.title("Odor 2 Action Collaboration Survey Data")

    # st.sidebar.markdown("""I talk or directly email with this person (for any reason)...\n""")

    # st.sidebar.markdown("""Graphs loading first plotting spread sheets...\n""")

    genre = st.sidebar.radio(
        "What's your prefered graph layout?",
        (
            "Hive",
            "Population",
            "Chord",
            "Physics",
            "Bundle",
            "Basic",
            "Spreadsheet",
            "AdjacencyMatrix",
        ),
    )

    st.sidebar.markdown(
        "Problem most people politely answer that they talk to someone a little bit, a bias if which is not corrected \n for hyperconnects everyone to everyone else in a meaningless way"
    )
    st.sidebar.markdown("solution threshold a meaningful level of communication")
    st.sidebar.markdown("The higher the threshold the more you reduce connections")

    threshold = st.slider("Select a threshold value", 0.0, 17.0, 5.0, 1.0)
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
    ) = get_frame(threshold)

    def get_table_download_link_csv(df):
        import base64

        # csv = df.to_csv(index=False)
        csv = df.to_csv().encode()
        # b64 = base64.b64encode(csv.encode()).decode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
        return href

    if genre == "Spreadsheet":
        st.markdown(get_table_download_link_csv(df2), unsafe_allow_html=True)
        st.markdown(get_table_download_link_csv(sheet), unsafe_allow_html=True)

        st.write(legend)
        st.write(df2)
        st.table(df2)
        # dl = st.radio(
        #    "Download?", ("No","Yes")# , "Bundle", "Basic","Spreadsheet")
        # )
        # if dl == "Yes":

    fig = plt.figure()
    for k, v in color_dict.items():
        plt.scatter([], [], c=v, label=k)
    plt.legend()
    fig.tight_layout()
    plt.axis("off")
    st.sidebar.pyplot(fig)
    inboth = set(names) & set(ratercodes)
    notinboth = set(names) - set(ratercodes)

    allcodes = set(names) or set(ratercodes)

    first = nx.DiGraph()

    for i, row in enumerate(allcodes):
        if i != 0:
            if row[0] != 1 and row[0] != 0:
                first.add_node(row[0], name=row)  # ,size=20)
    # popg = nx.DiGraph()
    adj_mat_dicts = []
    conns = {}
    cc = copy.copy(color_code_0)
    for i, idx in enumerate(df2.index):
        for j, col in enumerate(df2.columns):
            if col not in cc.keys():
                cc[col] = "Unknown"
            if idx not in color_code_0.keys():
                cc[idx] = "Unknown"

    for i, idx in enumerate(df2.index):
        for j, col in enumerate(df2.columns):
            weight = df2.iloc[i, j]
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
    # def dontdo(popg):
    if genre == "Population":
        fig, ax = plt.subplots(figsize=(20, 15))

        pos = nx.spring_layout(popg, k=15, seed=4572321, scale=1.5)
        sizes = {}
        for k, v in cc.items():
            if v not in sizes.keys():
                sizes[v] = 1
            else:
                sizes[v] += 1
        temp = list([s * 1000 for s in sizes.values()])
        # st.text(temp)
        # st.text(popg.node0s)
        node_color = [color_dict[n] for n in popg]
        nx.draw_networkx_nodes(
            popg,
            pos=pos,
            node_color=node_color,  # = [color_code[n] for n in H],
            node_size=temp,
            alpha=0.5,
            linewidths=2,
        )

        widths = []  # [e["weight"] for e in popg.edges]
        # st.text(widths)
        for e in popg.edges:
            e = popg.get_edge_data(cc[idx], cc[col])
            widths.append(e["weight"] * 0.025)
        nx.draw_networkx_edges(
            popg, pos=pos, edge_color="grey", alpha=0.5, width=widths
        )
        # labels = {v.name:v for v,v in popg.nodes}
        labels = {}
        for node in popg.nodes():
            # set the node name as the key and the label as its value
            labels[node] = node
        nx.draw_networkx_labels(popg, pos, labels, font_size=16, font_color="r")

        st.pyplot(fig)
    first.remove_nodes_from(list(nx.isolates(first)))
    edges_df_full = nx.to_pandas_adjacency(first)
    try:
        del edges_df_full["0"]
        del edges_df_full["1"]
    except:
        pass
    try:
        edges_df_full.drop("0", inplace=True)
        edges_df_full.drop("1", inplace=True)
    except:
        pass
    adj_mat = pd.DataFrame(adj_mat_dicts)
    encoded = {v: k for k, v in enumerate(first.nodes())}
    link = dict(
        source=[encoded[i] for i in list(adj_mat["src"].values)],
        target=[encoded[i] for i in list(adj_mat["tgt"].values)],
        value=[i * 3 for i in list(adj_mat["weight"].values)],
    )
    adj_mat2 = pd.DataFrame(link)
    adj_mat3 = adj_mat[adj_mat["weight"] != 0]
    if genre == "Physics":

        labels_ = st.radio(
            "Would you like node labels to be prominent, or degree size?",
            ("labels", "degsize"),
        )
        if labels_ == "labels":
            labels = True
        else:
            labels = False

        pos = nx.get_node_attributes(first, "pos")
        fig = plt.figure()
        d = nx.degree(first)
        temp = first.to_undirected()
        cen = nx.betweenness_centrality(temp)
        d = [((d[node] + 1) * 1.25) for node in first.nodes()]
        G = nx_G = first  # ead_graph()

        nt = Network(
            notebook=True,
            directed=True,
            height="750px",
            width="100%",
            font_color="black",  # , bgcolor='#222222'
        )  # bgcolor='#222222',
        # nt.toggle_hide_edges_on_drag(True)
        # nt.repulsion(300)

        nt.barnes_hut()
        nt.from_nx(G)
        # nt.nodes[3]['group'] = 10
        adj_mat = pd.DataFrame(adj_mat_dicts)

        edge_data = zip(
            list(adj_mat["src"].values),
            list(adj_mat["tgt"].values),
            list(adj_mat["weight"].values),
        )

        H = first.to_undirected()
        centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
        edge_thickness = {k: v * 20000 for v in centrality.items()}
        node_size = {k: v * 20000 for v in centrality.items()}

        for e in edge_data:
            src = e[0]
            dst = e[1]
            if labels:
                w = e[2]

            else:
                w = e[2] * 50

            # st.text(src)

            # nt.add_node(src, src, title=src,group=color_code[src])
            # nt.add_node(dst, dst, title=dst,group=color_code[src])
            nt.add_edge(src, dst, value=w)

        neighbor_map = nt.get_adj_list()

        # add neighbor data to node hover data
        for node in nt.nodes:
            if "title" not in node.keys():
                node["title"] = " Neighbors:<br>" + "<br>".join(
                    neighbor_map[node["id"]]
                )
            #
            if node["id"] in node_size.keys():
                if not labels:
                    node["size"] = 150 * node_size[node["id"]]
            # if node in node_size.keys():
            #    st.text(node_size[node])
            # st.text(node.keys())
            # st.text(node["id"])
            # node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
            node["label"] = str(node["id"])
            node["value"] = len(neighbor_map[node["id"]])
            node["color"] = color_code[node["id"]]
            if not labels:
                node["borderWidth"] = 10
                node["borderWidthSelected"] = 20

        if False:
            nt.show_buttons(filter_=["physics"])

        nt.show("test.html")

        HtmlFile = open("test.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()

        components.html(source_code, height=1100, width=1100)
        # st.markdown("Graphs below can be made to be interactive...")

    if genre == "Hive":

        # convert `networkx` edges and nodes into `hiveplotlib`-ready structures
        G = first
        encoded = {v: k for k, v in enumerate(first.nodes())}
        reverse = {v: k for k, v in encoded.items()}

        G = nx.relabel_nodes(G, encoded, copy=True)
        # for node in G.nodes:
        #    st.text(dir(node))
        #    st.text(type(node))
        #    node_ = encoded[node]
        #    G.nodes[node] = node_
        # node.encode()# = encoded[node]

        edges = np.array(G.edges)

        # pull out degree information from nodes for later use
        node_ids, degrees = np.unique(edges, return_counts=True)

        nodes = []
        Un_ind = []
        IRG1_indices = []
        IRG2_indices = []
        IRG3_indices = []
        DCMT_ind = []  # ,Un_ind

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

            else:
                # G.nodes.data()[node_id]['club'] = "Unknown"
                G.nodes.data()[node_id]["Unknown"] = 5
                G.nodes.data()[node_id]["club"] = 5

                Un_ind.append(i)

            temp_node = Node(unique_id=node_id, data=G.nodes.data()[node_id])
            nodes.append(temp_node)

        temp = list(set(color_code_0.values()))
        temp.append("Unknown")
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
            axes_names=temp,
            vmins=[0, 0, 0, 0, 0],
            vmaxes=[2, 2, 2, 2, 2],
            orient_angle=30,
        )

        # fig, ax = axes_viz_mpl(karate_hp,
        #                       axes_labels_buffer=1.4)

        # plot nodes
        # node_viz_mpl(hp,
        #             fig=fig, ax=ax, s=180, c="black")

        # change the line kwargs for edges in plot
        hp.add_edge_kwargs(
            axis_id_1=temp[0], axis_id_2=temp[1], c=f"C0", lw=0.1, alpha=1, zorder=10
        )
        hp.add_edge_kwargs(
            axis_id_1=temp[1], axis_id_2=temp[2], c=f"C2", lw=0.1, alpha=1, zorder=10
        )
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
        hp.place_nodes_on_axis(
            axis_id=temp[3],
            unique_ids=[nodes[i].data["loc"] for i in Un_ind],
            sorting_feature_to_use="loc",
            vmin=0,
            vmax=33,
        )

        hp.connect_axes(edges=edges, axis_id_1=temp[0], axis_id_2=temp[1], c="C0")
        hp.connect_axes(edges=edges, axis_id_1=temp[1], axis_id_2=temp[2], c="C1")
        hp.connect_axes(edges=edges, axis_id_1=temp[0], axis_id_2=temp[2], c="C3")
        hp.connect_axes(edges=edges, axis_id_1=temp[2], axis_id_2=temp[3], c="C4")
        hp.connect_axes(edges=edges, axis_id_1=temp[3], axis_id_2=temp[1], c="C5")
        hp.connect_axes(edges=edges, axis_id_1=temp[3], axis_id_2=temp[0], c="C6")
        hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[0], c="C7")
        hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[1], c="C8")
        hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[2], c="C9")
        hp.connect_axes(edges=edges, axis_id_1=temp[4], axis_id_2=temp[3], c="C10")

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

        st.pyplot(fig)

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

        def dontdo():
            """
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

    if genre == "Bundle":

        labels_ = st.radio("Would you like to label nodes?", ("No", "Yes"))
        if labels_ == "Yes":
            labels = True
        if labels_ == "No":
            labels = False
        fig4 = data_shade(first, color_code, adj_mat, color_dict, labels)
        st.pyplot(fig4)
    if genre == "Basic":
        labels_ = st.radio("Would you like to label nodes?", ("No", "Yes"))
        if labels_ == "Yes":
            labels_ = True
        if labels_ == "No":
            labels_ = False

        st.markdown(
            "for contrast see hair ball below (wiring length is not reduced)..."
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
        plt.legend()

        st.pyplot(fig)

    # from chord3 import doCircleRibbonGraph, get_colors
    # edges_df = networkx.to_pandas_adjacency(g)
    # labs = ["IRG1","IRG2","IRG3","DX"]
    # colors = get_colors(4)
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

    # fig = doCircleRibbonGraph(narr,labs, colors=ideo_colors, plot_size=500, title='Phd Country' )
    # st.write(fig)
    # chord = hv.Chord(adj_mat3)
    # st.write(pd.DataFrame(first.nodes))

    if genre == "AdjacencyMatrix":
        g = sns.clustermap(df2)
        st.pyplot(g)
        st.markdown("clustergram of adjacency matrix")

        columns = list(df2.columns.values)
        rows = list(df2.index)
        figure = dashbio.Clustergram(
            data=df2.loc[rows].values,
            column_labels=columns,
            color_threshold={"row": 250, "col": 700},
            hidden_labels="row",
            height=800,
            width=800,
        )
        # row_labels=list(df2.index.values),

        st.write(figure)

    if genre == "Chord":

        temp = pd.DataFrame(first.nodes)
        nodes = hv.Dataset(temp[0])

        links = copy.copy(adj_mat)
        links.rename(
            columns={"weight": "value", "src": "source", "tgt": "target"}, inplace=True
        )
        links = links[links["value"] != 0]
        # result = []
        # for k in links["source"]:
        #    result.append(color_code_0[k])

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
        for kk in df_links["source"]:
            k = df_nodes.loc[kk, "name"]
            # st.text(k)
            if k not in color_code_0.keys():
                color_code_0[k] = "Unknown"

        # Colors = [ (kk,color_code_0[df_nodes.loc[kk,'name']]) for kk in df_links["source"] ]
        # st.text(Colors)
        # df_links["colors"] = Colors
        # st.write(df_links)
        # st.write(df_nodes)

        pd.set_option("display.max_columns", 11)
        hv.extension("bokeh")
        hv.output(size=200)

        def dont():
            chord = hv.Chord(links)  # .select(value=(5, None))
            # node_color = [color_code[n] for n in H]
            # st.text(links['color'])
            chord.opts(
                opts.Chord(
                    cmap="Category20",
                    width=500,
                    height=500,
                    edge_cmap="Category20",
                    edge_color=dim("source").str(),
                    labels="name",
                    node_color=dim("index").str(),
                )
            )

        nodes = hv.Dataset(df_nodes, "index")
        # https://geomdata.gitlab.io/hiveplotlib/karate_club.html
        # Todo make hiveplot
        #
        chord = hv.Chord((df_links, nodes))  # .select(value=(5, None))
        chord.opts(
            opts.Chord(
                cmap="Category20",
                edge_cmap="Category20",
                edge_color=dim("colors").str(),
                width=500,
                height=500,
                labels="name",
                node_color=dim("colors").str(),
            )
        )
        # st.write(hv.render((chord), backend="bokeh"))
        # st.markdown("Chord layout democratic")
        hv.save(chord, "chord2.html", backend="bokeh")
        HtmlFile2 = open("chord2.html", "r", encoding="utf-8")
        source_code2 = HtmlFile2.read()
        components.html(source_code2, height=1100, width=1100)

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
