FROM continuumio/miniconda3

ADD ./requirements.txt /tmp/requirements.txt
ADD ./fgclassifier /app/fgclassifier
ADD ./config.py /app/config.py

WORKDIR /app

# Use Intel Python; Install dependencies
RUN conda config --add channels intel
RUN conda create -n idp intelpython3_core python=3
RUN conda activate idp
RUN pip install -qr /tmp/requirements.txt

# Prepare the environment more... (download nltk data, etc)
RUN python -m fgclassifier.prepare

# Start the app
CMD gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet -w 1 fgclassifier.visualizer:app
