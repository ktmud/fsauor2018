"""
Pre-train all models for later use
"""
import argparse
import itertools
import logging

from pathlib import Path
from config import model_save_path
from fgclassifier.train import train_and_save
from fgclassifier.visualizer.options import fm_choices, clf_choices
from fgclassifier.visualizer.actions import parse_model_choice

logger = logging.getLogger(__name__)


def train_all(skip_trained=False):
    trained = set()
    for fm, clf in itertools.product(fm_choices, clf_choices):
        lang, fm, clf, _ = parse_model_choice(fm, clf)
        sfx = '_en' if lang == 'en' else ''
        fname = f'{fm}_{clf}.pkl'
        trained.add(fname)
        if skip_trained or fname in trained:
            fpath = Path(model_save_path) / fname
            if fpath.exists():
                logger.info(f'Skipped {str(fpath)}.')
                continue
        logger.info(f'Training {fname}...')
        train_and_save(
            train_file=f'train{sfx}',
            valid_file=f'valid{sfx}',
            train_sample=5000,
            valid_sample=1000,
            feature_model=fm,
            classifier=clf,
            model='Baseline'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-trained', action='store_true',
                        help='Whether to skip trained models.')
    args = parser.parse_args()
    train_all(args.skip_trained)
