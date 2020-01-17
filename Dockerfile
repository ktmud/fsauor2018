# Use Intel Python
FROM python:3.7.6

WORKDIR /app

RUN apt-get update && apt-get install -y mariadb-client && rm -rf /var/lib/apt
RUN pip install pattern==3.6.0

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install scikit-learn==0.22.1 pandas==0.25.3 jieba==0.39
RUN pip install Flask==1.0.2 Flask-Cors==3.0.3 Flask-SocketIO==3.0.2 Flask-Assets==0.12
RUN pip install eventlet==0.24.1
RUN pip install TextBlob==0.15.2
RUN pip install snownlp==0.12.3
RUN pip install Flask-AutoIndex==0.6.2

# Install gensim
RUN pip install gensim==3.8.1
RUN pip install msgpack==0.6.2

# Install SpaCY language models
RUN pip install spacy==2.2.3
RUN python -m spacy download en_core_web_sm
RUN wget --quiet https://github.com/howl-anderson/Chinese_models_for_SpaCy/releases/download/v2.0.5/zh_core_web_sm-2.0.5.tar.gz
RUN pip install zh_core_web_sm-2.0.5.tar.gz && rm zh_core_web_sm-2.0.5.tar.gz

RUN pip install xgboost==0.81
RUN pip install joblib==0.14.1

# Prepare the environment more... (download nltk data)
ADD ./fgclassifier/prepare.py /tmp/prepare.py
RUN python /tmp/prepare.py && rm /tmp/prepare.py

# Add source code
ADD ./fgclassifier         /app/fgclassifier
ADD config.py CHECKS .env  /app/

# Start the app
CMD python -m fgclassifier.visualizer.app
