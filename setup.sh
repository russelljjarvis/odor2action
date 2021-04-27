sudo apt-get update
sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install seaborn
sudo python3 -m pip install plotly tabulate
sudo python3 -m conda install -c pyviz holoviews bokeh
sudo conda install -c pyviz holoviews bokeh
sudo python3 -m pip install git+https://github.com/pyviz/holoviews.git
git clone https://github.com/pyviz/holoviews.git
cd holoviews; sudo pip install -e .; cd ..;

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
