"""
The config

Note:
 Copy this file to config_local.py and change the path.
 Or, put the data files in directories as showing blow.
"""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
    # format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')

# When publicly deployed, this will be a different directory
data_root = os.environ.get("DATA_ROOT", "data")

model_save_path = f"{data_root}/models/"
train_data_path = f"{data_root}/train/sentiment_analysis_trainingset.csv"
valid_data_path = f"{data_root}/valid/sentiment_analysis_validationset.csv"
testa_data_path = f"{data_root}/test-a/sentiment_analysis_testa.csv"
testb_data_path = f"{data_root}/test-b/sentiment_analysis_testb.csv"

train_en_data_path = f"{data_root}/english_train.csv"
valid_en_data_path = f"{data_root}/english_valid.csv"
test_en_data_path = f"{data_root}/english_test.csv"

testa_predict_out_path = f"{data_root}/testa_prediction.csv"
testb_predict_out_path = f"{data_root}/testb_prediction.csv"


try:
    # import and override from local config
    from local_config import *
except ImportError:
    pass
