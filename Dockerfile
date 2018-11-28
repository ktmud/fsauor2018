# Use Intel Python
FROM intelpython/intelpython3_core:2019.1

WORKDIR /app

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install scikit-learn==0.20.1 pandas==0.23.4 jieba==0.39
RUN pip install Flask==1.0.2 Flask-Cors==3.0.3 Flask-SocketIO==3.0.2 Flask-Assets==0.12
RUN pip install eventlet==0.24.1 gunicorn==19.9.0
RUN pip install TextBlob==0.15.2
RUN pip install snownlp==0.12.3

# Install SpaCY language models
RUN pip install spacy==2.0.16
RUN wget --quiet https://github.com/howl-anderson/Chinese_models_for_SpaCy/releases/download/v2.0.5/zh_core_web_sm-2.0.5.tar.gz
RUN wget --quiet https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
RUN pip install zh_core_web_sm-2.0.5.tar.gz en_core_web_sm-2.0.0.tar.gz
RUN rm zh_core_web_sm-2.0.5.tar.gz en_core_web_sm-2.0.0.tar.gz

# Prepare the environment more... (download nltk data, etc)
ADD ./fgclassifier/prepare.py /tmp/prepare.py
RUN python /tmp/prepare.py

# Add source code
ADD ./fgclassifier /app/fgclassifier
ADD ./config.py /app/config.py

# Start the app
CMD gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet -w 1 fgclassifier.visualizer:app