FROM python:3.7.4
RUN apt-get update
RUN apt-get install -y python3-software-properties
RUN apt-get install -y software-properties-common
RUN apt-get -y install apt-transport-https ca-certificates
RUN apt-get install --fix-missing
ENV DISPLAY=:99
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null
RUN apt-get update
RUN apt-get install -y python3-dev
ADD requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt \
	&& rm -rf requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME

WORKDIR $APP_HOME/data

# --------------- Configure Streamlit ---------------
RUN mkdir -p /root/.streamlit
RUN touch /root/.streamlit/config.toml

RUN touch /root/.streamlit/credentials.toml
RUN echo "[general]" >> /root/.streamlit/credentials.toml
RUN echo 'email = "colouredstatic@gmail.com"' >> /root/.streamlit/credentials.toml

RUN bash -c 'echo -e "\
	[server]\n\
	enableCORS = false\n\
	enableXsrfProtection = false\n\
	\n\
	[browser]\n\
	serverAddress = \"0.0.0.0\"\
	" > /root/.streamlit/config.toml'

EXPOSE 8501


# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
