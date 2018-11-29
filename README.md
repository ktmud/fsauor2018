Fine-grained Sentiment Analysis on User Reviews
-----------------------------------------------

This is a solution for the [Fine-grained Sentiment Analysis of User Reviews](https://challenger.ai/competition/fsauor2018) challenge
from AI Challenger.

## Dataset

The original Chinese dataset can be [downloaded here](https://drive.google.com/file/d/1YYRWKJmahhVW7ZmzGeEtlKqDl4h-v0wG/view).

The English translations are already included in the repository.

## Getting Started

- Put data into the `data` folder, or
- `cp config.py config_local.py`, and edit the file paths in `config_local.py`.

### Train Models

Checkout the `notebooks` or

```bash
./fgclassifer/train.py -c LDA
```

### Visualize the Results

```
python app.py --port 5000
```

Change `--port` as your like.

## Deploy

The visualization can be easily deployed to via [Dokku](https://github.com/dokku/dokku).
Just make sure to upload your pre-trained models to the appropriate
[persistent storage](https://github.com/dokku/dokku/blob/master/docs/advanced-usage/persistent-storage.md)
directory on the host machine.

Here's a list of Dokku commands you can probably use:

```bash
alias dokku="ssh dokku@your-host"

git remote add dokku dokku@your-host/review-sentiments
git push dokku  # first push automatically creates the app

dokku config:set review-sentiments FLASK_SECRECT_KEY=`openssl rand -base64 16`
dokku config:set review-sentiments DATA_ROOT=/opt/storage

# For storing pre-trained models
dokku storage:mount review-sentiments /var/lib/dokku/data/storage/review-sentiments:/opt/storage
```

Then upload the dataset and pre-trained models to your host:

```
scp -r data/* /var/lib/dokku/data/storage/review-sentiments
```

## Local Development

I recommend using the Docker image:

```
docker build -t ktmud/fgclassifier .
docker-compose up
```

Note that `docker-compose` will add storage mapping between
your host machine and the Docker container, and set required
variables.

You need to have an `/opt/storage/` folder on your
host machine and make user it is accessible by Docker.

To run the app without Docker, install the required packages 
via `requirements.txt`, then make sure the data (and pre-trained models)
are in your `DATA_ROOT` (take a look at `config.py` for how file paths are
defined).

```
pip install -r requirement.txt
export DATA_ROOT="./data"
python fgclassifier/prepare.py
python app.py
```
