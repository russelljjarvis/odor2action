
sudo apt-get update
sudo apt-get upgrade
sudo python -m pip install -U pip
#sudo apt-get install graphviz graphviz-dev
#sudo conda install -c cython# bokeh graphviz_layout
sudo python3 -m pip install numpy --upgrade --ignore-installed
sudo python3 -m pip install cython --upgrade --ignore-installed


sudo python3 -m pip install -r requirements.txt
sudo conda install -c pyviz scikit-image# bokeh graphviz_layout
sudo python3 -m conda install -c pyvis bokeh seaborn dash_bio scikit-image
sudo python3 -m pip install seaborn bokeh# holoviews==1.14.1
sudo python3 -m pip install pyvis dash_bio cython scikit-image
sudo python3 -m pip install dask plotly tabulate bokeh hiveplotlib hiveplot pygraphviz#==2.0.0#2.2
sudo python3 -m pip install streamlit --upgrade --ignore-installed
sudo pip install hiveplotlib pygraphviz
# streamlit-agraph
sudo $(which pip) install git+https://github.com/taynaud/python-louvain.git@networkx2
git clone https://github.com/taynaud/python-louvain.git@networkx2; cd networkx; sudo python3 setup.py install; cd -
python3 -c "from holoviews.operation.datashader import datashade"
git clone https://github.com/pygraphviz/pygraphviz; cd pygraphviz; sudo python3 setup.py install; cd -
#sudo python3 -m pip install git+https://github.com/pyviz/holoviews.git
#git clone https://github.com/pyviz/holoviews.git
#cd holoviews; sudo pip install -e .; cd ..;
#sudo python3 -m pip install pygraphviz
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
