
"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)
"""

import argparse
import numpy as np
import networkx as nx
#import node2vec
from node2vec import node2vec
from gensim.models import Word2Vec
from node2vec.edges import HadamardEmbedder
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
from holoviews import opts, dim
from collections import Iterable
import networkx

import holoviews as hv
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

from chord import Chord

import dash_bio
#def disable_logo(plot, element):
#	plot.state.toolbar.logo = None


#hv.extension("bokeh", logo=False)
#hv.output(size=150)
#hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
import plotly.graph_objects as go
from auxillary_methods import plotly_sized2#, data_shade#, draw_wstate_tree
#@st.cache(suppress_st_warning=True)

from datashader.bundling import hammer_bundle

from typing import List
import pandas as pd
import holoviews as hv
#from holoviews import opts, dim
#from bokeh.sampledata.les_mis import data

def generate_sankey_figure(nodes_list: List, edges_df: pd.DataFrame,
							   title: str = 'Sankey Diagram'):



	edges_df['src'] = edges_df['src'].apply(lambda x:
													nodes_list.index(x))
	edges_df['tgt'] = edges_df['tgt'].apply(lambda x:
													nodes_list.index(x))
	# creating the sankey diagram
	data = dict(
		type='sankey',
		node=dict(
			hoverinfo="all",
			pad=15,
			thickness=20,
				line=dict(
				color="black",
				width=0.5
			),
			label=nodes_list,
		),
		link=dict(
			source=edges_df['src'],
			target=edges_df['tgt'],
			value=edges_df['weight']
		)
	)

	layout = dict(
		title=title,
		font=dict(
			size=10
		)
	)

	fig = go.Figure(data=[data], layout=layout)
	st.write(fig)

	#return fig
def data_shade(graph):

	nodes = graph.nodes
	#orig_pos=nx.get_node_attributes(graph,'pos')

	nodes_ind = [i for i in range(0,len(graph.nodes()))]
	redo  = {k:v for k,v in zip(graph.nodes,nodes_ind)}

	pos_= nx.spring_layout(graph,scale=125, k=0.15, seed=4572321))
	#node_color = [community_index[n] for n in graph]
	H = graph.to_undirected()
	centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
	node_size = [v * 20000 for v in centrality.values()]

	coords = []
	for node in graph.nodes:
		 x, y = pos_[node]
		 coords.append((x, y))
	nodes_py = [[new_name, pos[0], pos[1]] for name, pos,new_name in zip(nodes, coords,nodes_ind)]
	ds_nodes = pd.DataFrame(nodes_py, columns=["name", "x", "y"])
	ds_edges_py = []
	for (n0, n1) in graph.edges:
		ds_edges_py.append([redo[n0], redo[n1]])
	ds_edges = pd.DataFrame(ds_edges_py, columns=["source", "target"])
	hb = hammer_bundle(ds_nodes, ds_edges)
	hbnp = hb.to_numpy()
	splits = (np.isnan(hbnp[:,0])).nonzero()[0]
	start = 0
	segments = []
	for stop in splits:
		 seg = hbnp[start:stop, :]
		 segments.append(seg)
		 start = stop


	fig, ax = plt.subplots(figsize=(15,15))

	for seg in segments:
		 ax.plot(seg[:,0], seg[:,1])

	ax3 = nx.draw_networkx_nodes(graph, pos_, node_size=node_size, node_shape='o', alpha=0.5, vmin=None, vmax=None, linewidths=1.0, label=None,ax=ax)#, **kwds)

	return fig
import seaborn as sns;
def plot_stuff(df2,edges_df_full,first,adj_mat_dicts):
	with shelve.open("fast_graphs_splash.p") as db:
		flag = 'chord' in db
		if False:#flag:
			graph = db['graph']
			#graph.opts(
			#	color_index="circle",
			#	width=150,
			#	height=150,
			#	show_frame=False,
			#	xaxis=None,
			#	yaxis=None,
			#	tools=["hover", "tap"],
			#	node_size=10,
			#	cmap=["blue", "orange"],
			#)
			#st.write(hv.render(graph, backend="bokeh"))


			chord = db['chord']
			st.write(chord)

		else:
			#graph = hv.Graph.from_networkx(
			#	first, networkx.layout.fruchterman_reingold_layout
			#)
			#graph.opts(
			#	color_index="circle",
			#	width=350,
			#	height=350,
			#	show_frame=False,
			#	xaxis=None,
			#	yaxis=None,
			#	tools=["hover", "tap"],
			#	node_size=10,
			#	cmap=["blue", "orange"],
			#)
			#st.write(hv.render(graph, backend="bokeh"))

			#nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
			#edges = [
			#    (0, 1, 53), (0, 2, 47), (2, 6, 17), (2, 3, 30), (3, 1, 22.5), (3, 4, 3.5), (3, 6, 4.), (4, 5, 0.45)
			#]

			#db['graph'] = graph

			chord = chord2.make_filled_chord(edges_df_full)
			st.write(chord)
			db['chord'] = chord
			adj_mat = pd.DataFrame(adj_mat_dicts)
			encoded = {v:k for k,v in enumerate(first.nodes())}
			link = dict(source = [encoded[i] for i in list(adj_mat["src"].values)][0:30], target =[encoded[i] for i in list(adj_mat["tgt"].values)][0:30], value =[i*3 for i in list(adj_mat["weight"].values)][0:30])
			data = go.Sankey(link = link)
			#fig = go.Figure(data)
			#st.write(fig)
			#fig.show(renderer="svg", width=1000, height=500)
			#fig.savefig('blah.svg')
			#db['sankey'] = data
			edge_list = networkx.to_edgelist(first)
			labels = list(first.nodes)
			#adjdf = nx.to_pandas_adjacency(first)
			#to_pandas_adjacency
			edge_list = nx.to_edgelist(first)
			columns = list(df2.columns.values)
			rows = list(df2.index[1:-1])
			figure=dashbio.Clustergram(
			        data=df2.loc[rows].values,
			        column_labels=columns,
			        row_labels=rows,
			        color_threshold={
			            'row': 250,
			            'col': 700
			        },
			        hidden_labels='row',
			        height=800,
			        width=800
			    )
			st.write(figure)
			#hv.Chord(edge_list,label=labels)
			g = sns.clustermap(df2)
			st.pyplot(g)
			plot_imshow_plotly(df2)

			#chord3 = chord2.make_filled_chord(adj_mat)
			#st.write(chord3)
			#db['chord3'] = chord3

		db.close()

def get_frame(new = True):

	with shelve.open("fast_graphs_splash.p") as store:
		flag = 'df' in store
		if False:
			df = store['df']  # load it

			df2 = store['df2']  # load it
			names = store['names']# = names  # save it
			ratercodes = store['ratercodes']# =   # save it
			legend = store['legend']# = legend  # save it

		else:
			if new:
				xlsx_file = Path('o2anetmap2021.xlsx')
			else:
				xlsx_file = Path('o2anetmap.xlsx')
			wb_obj = openpyxl.load_workbook(xlsx_file)

			# Read the active sheet:
			worksheet = wb_obj.active
			df2 = pd.DataFrame(worksheet.values)
			df = pd.DataFrame(worksheet.values)

			col_to_rename = df2.columns


			ratercodes = df[0][1:-1]
			row_names = df.T[0].values
			row_names = row_names[2:-1]
			names = [ rn.split("- ") for rn in row_names ]
			names2 = []
			for i in names :
				if len(i)==2:
					names2.append(i[1])
				else:
					names2.append(i)
			names = names2

			row_names = range(1, 114, 1)
			to_rename = {k:v for k,v in zip(row_names,names)}


			r_names = df.index.values[1:-1]
			to_rename_ind = {v:k for k,v in zip(df2[0][1:-1],r_names)}
			del df2[0]
			del df2[1]
			del df2[112]
			del df2[113]
			del df2[42]
			df2.drop(0,inplace=True)
			df2.drop(1,inplace=True)
			df2.drop(42,inplace=True)

			df2.rename(columns=to_rename,inplace=True)
			df2.rename(index=to_rename_ind,inplace=True)
			legend = {}
			legend.update({'Never':0.0})
			legend.update({'Barely or never':1})
			legend.update({'Occasionally in a minor way':2})
			legend.update({'Less than once a month':3})
			legend.update({'More than once a month (But not weekly)':4})
			legend.update({'Occasionally but substantively':5})
			legend.update({'More than twice a week':6})
			legend.update({'Often':7})
			legend.update({'Much or all of the time':8})
			legend.update({'1-2 times a week':9.0})

			df2.replace({'Never':0.0},inplace=True)
			df2.replace({'Barely or never':1},inplace=True)
			df2.replace({'Occasionally in a minor way':2},inplace=True)
			df2.replace({'Less than once a month':3},inplace=True)
			df2.replace({'More than once a month (But not weekly)':4},inplace=True)
			df2.replace({'Occasionally but substantively':5},inplace=True)
			df2.replace({'More than twice a week':6},inplace=True)
			df2.replace({'Often':7},inplace=True)
			df2.replace({'Much or all of the time':8},inplace=True)
			df2.replace({'1-2 times a week':9.0},inplace=True)
			store['df2'] = df2  # save it
			store['df'] = df  # save it

			store['names'] = names  # save it
			store['ratercodes'] = ratercodes  # save it
			store['legend'] = legend  # save it
			#fig = go.Figure(data)
			#st.write(fig)

	return df,df2,names,ratercodes,legend
#import networkx as nx
import networkx

sns_colorscale = [[0.0, '#3f7f93'], #cmap = sns.diverging_palette(220, 10, as_cmap = True)
	[0.071, '#5890a1'],
	[0.143, '#72a1b0'],
	[0.214, '#8cb3bf'],
	[0.286, '#a7c5cf'],
	[0.357, '#c0d6dd'],
	[0.429, '#dae8ec'],
	[0.5, '#f2f2f2'],
	[0.571, '#f7d7d9'],
	[0.643, '#f2bcc0'],
	[0.714, '#eda3a9'],
	[0.786, '#e8888f'],
	[0.857, '#e36e76'],
	[0.929, '#de535e'],
	[1.0, '#d93a46']]


def df_to_plotly(df,log=False):
	return {'z': df.values.tolist(),
			'x': df.columns.tolist(),
			'y': df.index.tolist()}


def plot_df_plotly(sleep_df):
	fig = go.Figure(data=go.Heatmap(df_to_plotly(sleep_df,log=True)))
	st.write(fig)

def plot_imshow_plotly(sleep_df):

	heat = go.Heatmap(df_to_plotly(sleep_df),colorscale=sns_colorscale)
	#fig = go.Figure(data=

	title = 'Adjacency Matrix'

	layout = go.Layout(title_text=title, title_x=0.5,
					width=600, height=600,
					xaxis_showgrid=False,
					yaxis_showgrid=False,
					yaxis_autorange='reversed')

	fig=go.Figure(data=[heat], layout=layout)

	st.write(fig)


def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)

	return
#from streamlit import components

def main():
	st.title('NeuroScience Collaboration Survey Data')

	#st.text(dir(nx))
	st.markdown("""I talk or directly email with this person (for any reason)...\n""")

	st.markdown("""Graphs loading first plotting spread sheets...\n""")

	option = st.checkbox('Last year?')
	if option:
		df,df2,names,ratercodes,legend = get_frame(new=False)
	else:
		df,df2,names,ratercodes,legend = get_frame(new=True)

	# https://coderzcolumn.com/tutorials/data-science/how-to-plot-chord-diagram-in-python-holoviews

	#st.sidebar.title('Choose your favorite Graph')
	#option=st.selectbox('select graph',('Simple','Karate', 'GOT'))
	option = st.checkbox('consult spread sheet?')
	"""
	Note clicking yes wont result in instaneous results
	please scroll down to explore putative network visualizations
	"""
	if option:
		st.write(df)
		st.write(legend)
		st.write(df2)
	st.markdown("""Still loading Graphs please wait...\n""")

	inboth = set(names) & set(ratercodes)
	notinboth = set(names) - set(ratercodes)

	allcodes = set(names) or set(ratercodes)


	first = nx.DiGraph()
	for i,row in enumerate(allcodes):
		if i!=0:
			if row[0]!=1 and row[0]!=0:
				first.add_node(row[0],name=row)#,size=20)

	adj_mat_dicts = []
	for idx in df2.index:
		for col in df2.columns:
			if idx != col:
				try:
					weight = df2.loc[idx, col][0]
				except:
					weight = df2.loc[idx, col]
				adj_mat_dicts.append({"src":idx,"tgt":col,"weight":weight})
				#print(adj_mat_dicts[-1])
				#adj_mat_dicts.append({"src":idx,"tgt":col,"weight":weight})

				first.add_edge(idx,col,weight=weight)

	#nt = Network("500px", "500px",notebook=True,heading='')
	#nt.from_nx(first)
	#st.text(dir(nt))
	#nt.show()
	#nt.write_html()

	#import streamlit
	#from streamlit_agraph import agraph, Node, Edge, Config
	#config = Config(width=500,
	#          height=500,
	#           directed=True,
	#            nodeHighlightBehavior=True,
	#             highlightColor="#F7A7A6", # or "blue"
	#              collapsible=True,
	# coming soon (set for all): node_size=1000, node_color="blue"
	#               )

	#return_value = agraph(nodes=first.nodes,
	#                      edges=first.edges,
	#                      config=config)

	matrix = df2.to_numpy()
	names = list(first.nodes())
	first.remove_nodes_from(list(nx.isolates(first)))
	#st.text(type(first))
	try:
		edges_df_full = nx.to_pandas_adjacency(first)
	except:
		edges_df_full = nx.to_pandas_dataframe(first)
	#st.write(edges_df_full)
	try:
		del edges_df_full["0"]
		del edges_df_full["1"]
	except:
		pass
	try:
		edges_df_full.drop("0",inplace=True)
		edges_df_full.drop("1",inplace=True)
	except:
		pass

	pos = nx.get_node_attributes(first,'pos')
	#assert len(gro_pos)==len(micro_gro.nodes)
	fig = plt.figure()

	d = nx.degree(first)

	temp = first.to_undirected()
	cen = nx.betweenness_centrality(temp)
	#st.text("who are the research hubs?")
	#for k,v in zip(list(first.nodes),list(cen.values())):
	#	st.text(str(k)+" degree"+str(v))
	d = [((d[node]+1) * 1.25) for node in first.nodes()]
	G = nx_G = first#ead_graph()

	nt = Network("500px", "500px",notebook=True,heading='Elastic Physics Network Survey Data')
	nt.barnes_hut()
	nt.from_nx(G)
	nt.show_buttons(filter_=['physics'])
	nt.show('test.html')

	HtmlFile = open("test.html", 'r', encoding='utf-8')
	source_code = HtmlFile.read()
	try:
		components.v1.html(source_code, height = 1100,width=1100)
	except:
		components.html(source_code, height = 1100,width=1100)
	st.text("keep scrolling down...")
	# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
	#n2vec = node2vec.Node2Vec(nx_G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
	# Embed nodes
	#model = n2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
	# Look for most similar nodes
	#model.wv.most_similar(list(nx_G.nodes)[0])  # Output node names are always strings
	#edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

		# https://nbviewer.jupyter.org/github/ykhorram/nips2015_topic_network_analysis/blob/master/nips_collaboration_network.ipynb
	#except:
	#pos = nx.spring_layout(first, scale=4.5)
	#if 'graphviz_layout' in locals():
	#	pos = graphviz_layout(first)


	#ax1 = nx.draw_networkx_nodes(first,pos,node_size=d, node_shape='o', alpha=0.35, label=None)
	#ax01 = nx.draw_networkx_edges(first,pos, width=0.25, edge_color='blue', style='solid', alpha=0.35,arrows=False, label=None)
	#st.pyplot(fig)
	H = first.to_undirected()
	centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
	#centrality = nx.betweenness_centrality(H), endpoints=True)

	# compute community structure
	lpc = nx.community.label_propagation_communities(H)
	community_index = {n: i for i, com in enumerate(lpc) for n in com}

	#### draw graph ####
	fig, ax = plt.subplots(figsize=(20, 15))
	pos = nx.spring_layout(H, k=0.15, seed=4572321)
	node_color = [community_index[n] for n in H]
	node_size = [v * 20000 for v in centrality.values()]
	nx.draw_networkx(
	    H,
	    pos=pos,
	    with_labels=False,
	    node_color=node_color,
	    node_size=node_size,
	    edge_color="gainsboro",
	    alpha=0.4,
	)

	# Title/legend
	font = {"color": "k", "fontweight": "bold", "fontsize": 20}
	ax.set_title("network", font)
	# Change font color for legend
	font["color"] = "b"

	#ax.text(
	#    0.80,
	#    0.10,
	#    "node color = community structure",
	#    horizontalalignment="center",
	#    transform=ax.transAxes,
	#    fontdict=font,
	#)
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
	st.pyplot(fig)

	plot_stuff(df2,edges_df_full,first,adj_mat_dicts)
	fig4 = data_shade(first)
	st.pyplot(fig4)

	def dontdo():
		fig4 = data_shade(first)
		st.pyplot(fig4)

		adj_mat = pd.DataFrame(adj_mat_dicts)
		link = dict(source = adj_mat["src"], target = adj_mat["tgt"], value = adj_mat["weight"])




		#generate_sankey_figure(list(first.nodes), adj_mat,title = 'Sankey Diagram')

		fig = go.Figure(data=[go.Sankey(
			node = dict(
			  pad = 15,
			  thickness = 20,
			  line = dict(color = "black", width = 0.5),
			  label = list(first.nodes()),#["A1", "A2", "B1", "B2", "C1", "C2"],
			  color = "blue"
			),
			link = dict(source = adj_mat["src"], target = adj_mat["tgt"], value = [i*10 for i in adj_mat["weight"]]))])

		fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)

		st.write(fig)
		link = dict(source = adj_mat["src"], target = adj_mat["tgt"], value = [i*10 for i in adj_mat["weight"]])

		data = go.Sankey(link = link)

		#fig3 = go.Figure(data)
		layout = go.Layout(
			paper_bgcolor="rgba(0,0,0,0)",  # transparent background
			plot_bgcolor="rgba(0,0,0,0)",  # transparent 2nd background
			xaxis={"showgrid": False, "zeroline": False},  # no gridlines
			yaxis={"showgrid": False, "zeroline": False},  # no gridlines
		)  # Create figure
		layout["width"] = 925
		layout["height"] = 925

		fig3 = go.Figure(data,layout=layout)  # Add all edge traces
		st.write(fig3)

	#nodes = first.nodes
	#edges = first.edges
	#value_dim = [i*10 for i in adj_mat["weight"]]
	#careers = hv.Sankey((edges, nodes), ['From', 'To'])#, vdims=value_dim)

	#careers.opts(
	#    opts.Sankey(labels='label', label_position='right', width=900, height=300, cmap='Set1',
	#                edge_color=dim('To').str(), node_color=dim('index').str()))
	#careers.write(hv.render(graph, backend="bokeh"))

	#for trace in edge_trace:
	#	fig.add_trace(trace)  # Add node trace
	#fig.add_trace(node_trace)  # Remove legend

	#fig.show()
if __name__ == "__main__":

	main()
