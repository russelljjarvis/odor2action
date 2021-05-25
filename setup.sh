
apt-get update
apt-get upgrade
#apt-get install -y graphviz #| grep http | awk '{print $1}' | tr -d "'"
# git clone https://github.com/pygraphviz/pygraphviz; cd pygraphviz; python3 setup.py install; cd -

# Create .profile.d script (sourced by dyno) setting PATH and LD_LIBRARY_PATH
#mkdir -p .profile.d
#echo "PATH=/app/$install_dir/usr/bin:\$PATH" >.profile.d/graphviz.sh
#echo "export LD_LIBRARY_PATH=/app/$libs_dir:\$LD_LIBRARY_PATH" >>.profile.d/graphviz.sh

# Run dot -c to configure plugins (creates config6a file)
#LD_LIBRARY_PATH=$libs_dir "$install_dir"/usr/bin/dot -c


python -m pip install -U pip
#apt-get install graphviz graphviz-dev
#conda install -c cython# bokeh graphviz_layout
python3 -m pip install numpy --upgrade --ignore-installed
python3 -m pip install cython --upgrade --ignore-installed

python3 -m pip install -r requirements.txt
#conda install -c pyviz scikit-image# bokeh graphviz_layout
#python3 -m conda install -c pyvis bokeh seaborn scikit-image # dash_bio
python3 -m pip install bokeh# holoviews==1.14.1 seaborn
python3 -m pip install pyvis cython scikit-image #dash_bio dask
python3 -m pip install plotly tabulate # hiveplotlib hiveplot pygraphviz#==2.0.0#2.2
python3 -m pip install streamlit --upgrade --ignore-installed
#python3 -m pip install pygraphviz
# streamlit-agraph
$(which pip) install python-igraph

#$(which python) -m pip install git+https://github.com/taynaud/python-louvain.git@networkx2
#python3 -m pip install git+https://github.com/taynaud/python-louvain.git@networkx2

python3 make_serial_plots0.py
python3 make_serial_plots1.py

$(which pip) install git+https://github.com/taynaud/python-louvain.git@networkx2
git clone https://github.com/taynaud/python-louvain.git@networkx2; cd networkx2; $(which python) setup.py install; cd -

#python3 -c "from holoviews.operation.datashader import datashade"
#python3 -m pip install git+https://github.com/pyviz/holoviews.git
#git clone https://github.com/pyviz/holoviews.git
#cd holoviews; pip install -e .; cd ..;
#python3 -m pip install pygraphviz


mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"rjjarvis@asu.edu\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
