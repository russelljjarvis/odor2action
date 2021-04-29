
sudo apt-get update
#sudo apt-get install graphviz graphviz-dev
sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn
sudo python3 -m pip install bs4
sudo python3 -m pip install natsort dask plotly tabulate streamlit-agraph
#git clone https://github.com/pygraphviz/pygraphviz; cd pygraphviz; sudo python3 setup.py install; cd -
#sudo python3 -m conda install -c pyviz holoviews bokeh seaborn
sudo conda install -c pyviz holoviews# bokeh graphviz_layout
sudo python3 -m pip install git+https://github.com/pyviz/holoviews.git
git clone https://github.com/pyviz/holoviews.git
cd holoviews; sudo pip install -e .; cd ..;
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
