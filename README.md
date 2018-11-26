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

### Check Visualizations

```
python app.py --port 5000
```

Change `--port` as your like.
