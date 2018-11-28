# Use Intel Python
FROM intelpython/intelpython3_core:2019.1

WORKDIR /app

# Install pip dependencies
RUN wget --quiet https://github.com/howl-anderson/Chinese_models_for_SpaCy/releases/download/v2.0.5/zh_core_web_sm-2.0.5.tar.gz
RUN wget --quiet https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm zh_core_web_sm-2.0.5.tar.gz en_core_web_sm-2.0.0.tar.gz

# Prepare the environment more... (download nltk data, etc)
ADD ./fgclassifier/prepare.py /tmp/prepare.py
RUN python /tmp/prepare.py

# Add source code
ADD ./fgclassifier /app/fgclassifier
ADD ./config.py /app/config.py


# Start the app
CMD gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet -w 1 fgclassifier.visualizer:app
