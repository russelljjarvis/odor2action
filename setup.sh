
apt-get update
apt-get upgrade
python -m pip install -U pip
#apt-get install graphviz graphviz-dev
#conda install -c cython# bokeh graphviz_layout
python3 -m pip install numpy --upgrade --ignore-installed
python3 -m pip install cython --upgrade --ignore-installed
$(which python) -m pip install git+https://github.com/taynaud/python-louvain.git@networkx2
python3 -m pip install git+https://github.com/taynaud/python-louvain.git@networkx2

python3 -m pip install -r requirements.txt
conda install -c pyviz scikit-image# bokeh graphviz_layout
python3 -m conda install -c pyvis bokeh seaborn dash_bio scikit-image
python3 -m pip install seaborn bokeh# holoviews==1.14.1
python3 -m pip install pyvis dash_bio cython scikit-image
python3 -m pip install dask plotly tabulate bokeh hiveplotlib hiveplot pygraphviz#==2.0.0#2.2
python3 -m pip install streamlit --upgrade --ignore-installed
pip install hiveplotlib pygraphviz
# streamlit-agraph
$(which pip) install python-igraph
$(which pip) install git+https://github.com/taynaud/python-louvain.git@networkx2
git clone https://github.com/taynaud/python-louvain.git@networkx2; cd networkx2; $(which python) setup.py install; cd -
python3 -c "from holoviews.operation.datashader import datashade"
git clone https://github.com/pygraphviz/pygraphviz; cd pygraphviz; python3 setup.py install; cd -
#python3 -m pip install git+https://github.com/pyviz/holoviews.git
#git clone https://github.com/pyviz/holoviews.git
#cd holoviews; pip install -e .; cd ..;
#python3 -m pip install pygraphviz
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"russelljarvis@protonmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
