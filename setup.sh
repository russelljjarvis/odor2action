
sudo apt-get update
sudo apt-get upgrade
sudo python -m pip install -U pip
#sudo apt-get install graphviz graphviz-dev
#sudo conda install -c cython# bokeh graphviz_layout
sudo python3 -m pip install numpy --upgrade --ignore-installed
sudo python3 -m pip install cython --upgrade --ignore-installed

sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn holoviews bokeh
sudo python3 -m pip install pyvis dash_bio cython scikit-image
sudo python3 -m pip install dask plotly tabulate# streamlit-agraph
#git clone https://github.com/pygraphviz/pygraphviz; cd pygraphviz; sudo python3 setup.py install; cd -
sudo python3 -m conda install -c pyvis holoviews bokeh seaborn dash_bio scikit-image
sudo conda install -c pyviz holoviews scikit-image# bokeh graphviz_layout
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
