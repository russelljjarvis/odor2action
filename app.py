	#!/usr/bin/env python
# coding: utf-8


"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)
"""

import shelve
import streamlit as st
import os

import pandas as pd
import pickle
import streamlit as st
from holoviews import opts, dim
from collections import Iterable
import networkx

#from auxillary_methods import author_to_coauthor_network, network#,try_again
import holoviews as hv
#from auxillary_methods import push_frame_to_screen, plotly_sized#, data_shade, draw_wstate_tree
import chord2
import shelve

import plotly.graph_objects as go
import pandas as pd
#import geopandas
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


def disable_logo(plot, element):
	plot.state.toolbar.logo = None


hv.extension("bokeh", logo=False)
hv.output(size=150)
hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
import plotly.graph_objects as go
#from auxillary_methods import plotly_sized#, data_shade#, draw_wstate_tree
#@st.cache(suppress_st_warning=True)

from datashader.bundling import hammer_bundle

from typing import List

#@staticmethod
def generate_sankey_figure(nodes_list: List, edges_df: pd.DataFrame,
							   title: str = 'Sankey Diagram'):



	# Create the node indices
	#nodes_list = nodes_df['Node'].tolist()
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

	pos_= nx.spring_layout(graph,scale=125)
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

	ax3 = nx.draw_networkx_nodes(graph, pos_, node_size=25, node_shape='o', alpha=0.5, vmin=None, vmax=None, linewidths=1.0, label=None,ax=ax)#, **kwds)

	return fig
# data_shade(second,world,colors)

def plot_stuff(df2,edges_df_full,first):
	with shelve.open("fast_graphs_splash.p") as db:
		flag = 'chord' in db
		if False:#flag:
			graph = db['graph']
			graph.opts(
				color_index="circle",
				width=150,
				height=150,
				show_frame=False,
				xaxis=None,
				yaxis=None,
				tools=["hover", "tap"],
				node_size=10,
				cmap=["blue", "orange"],
			)
			st.write(hv.render(graph, backend="bokeh"))


			chord = db['chord']
			st.write(chord)

		else:
			graph = hv.Graph.from_networkx(
				first, networkx.layout.fruchterman_reingold_layout
			)
			graph.opts(
				color_index="circle",
				width=350,
				height=350,
				show_frame=False,
				xaxis=None,
				yaxis=None,
				tools=["hover", "tap"],
				node_size=10,
				cmap=["blue", "orange"],
			)
			st.write(hv.render(graph, backend="bokeh"))

			#nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
			#edges = [
			#    (0, 1, 53), (0, 2, 47), (2, 6, 17), (2, 3, 30), (3, 1, 22.5), (3, 4, 3.5), (3, 6, 4.), (4, 5, 0.45)
			#]

			db['graph'] = graph

			chord = chord2.make_filled_chord(edges_df_full)
			st.write(chord)
			db['chord'] = chord

			#chord3 = chord2.make_filled_chord(adj_mat)
			#st.write(chord3)
			#db['chord3'] = chord3

		db.close()

def get_frame():

	with shelve.open("fast_graphs_splash.p") as store:
		flag = 'df' in store
		if flag:
			df = store['df']  # load it

			df2 = store['df2']  # load it
			names = store['names']# = names  # save it
			ratercodes = store['ratercodes']# =   # save it
			legend = store['legend']# = legend  # save it

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

	return df,df2,names,ratercodes,legend


def main():
	st.markdown("""I talk or directly email with this person (for any reason)...\n""")

	st.markdown("""Graphs loading first plottin spread sheets...\n""")


	df,df2,names,ratercodes,legend = get_frame()
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
	adj_mat = pd.DataFrame(adj_mat_dicts)
	first.remove_nodes_from(list(nx.isolates(first)))
	#first = nx.remove_isolated(first)
	edges_df_full = networkx.to_pandas_adjacency(first)
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
	d = [((d[node]+1) * 1.25) for node in first.nodes()]
	#nx.draw(first,node_size=d)
	pos = nx.spring_layout(first, scale=4.5)
	ax1 = nx.draw_networkx_nodes(first,pos,node_size=d, node_shape='o', alpha=0.35, label=None)
	#ax0 = nx.draw_networkx_nodes(micro_gro, gro_pos,node_size=5, node_color='grey', node_shape='o', alpha=0.35, width=0.1, label=None)
	ax01 = nx.draw_networkx_edges(first,pos, width=0.25, edge_color='blue', style='solid', alpha=0.35,arrows=False, label=None)
	st.pyplot(fig)
	#st.write(edges_df_full)
	plot_stuff(df2,edges_df_full,first)

	def dontdo():
		fig4 = data_shade(first)
		st.pyplot(fig4)

		adj_mat = pd.DataFrame(adj_mat_dicts)
		link = dict(source = adj_mat["src"], target = adj_mat["tgt"], value = adj_mat["weight"])



		#fig0 = plotly_sized(first)
		#st.write(fig0)

		generate_sankey_figure(list(first.nodes), adj_mat,title = 'Sankey Diagram')

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
		#fig.show()

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
