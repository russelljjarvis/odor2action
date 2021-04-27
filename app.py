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



import pandas as pd
import openpyxl
from pathlib import Path
import numpy as np
import networkx as nx

import xlrd


def disable_logo(plot, element):
	plot.state.toolbar.logo = None


hv.extension("bokeh", logo=False)
hv.output(size=350)
hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

#@st.cache(suppress_st_warning=True)
def plot_stuff(df2,edges_df_full,first):
	with shelve.open("fast_graphs_splash.p") as db:
		flag = 'chord' in db
		if flag:
			graph = db['graph']
			graph.opts(
				color_index="circle",
				width=250,
				height=250,
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
				width=250,
				height=250,
				show_frame=False,
				xaxis=None,
				yaxis=None,
				tools=["hover", "tap"],
				node_size=10,
				cmap=["blue", "orange"],
			)
			st.write(hv.render(graph, backend="bokeh"))
			db['graph'] = graph

			chord = chord2.make_filled_chord(edges_df_full)
			st.write(chord)
			db['chord'] = chord
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
			names2 = []#[i[1]
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
			to_rename_ind;


			del df2[0]
			del df2[1]

			del df2[112]
			del df2[113]
			del df2[42]

			df2.drop(0,inplace=True)
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
			first.add_node(row[0],name=row)


	for idx in df2.index:
		for col in df2.columns:
			if idx != col:
				try:
					weight = df2.loc[idx, col][0]
				except:
					weight = df2.loc[idx, col]

				first.add_edge(idx,col,weight=weight)



	edges_df_full = networkx.to_pandas_adjacency(first)
	del edges_df_full["0"]
	del edges_df_full["1"]
	edges_df_full.drop("0",inplace=True)
	edges_df_full.drop("1",inplace=True)
	plot_stuff(df2,edges_df_full,first)

if __name__ == "__main__":

    main()
