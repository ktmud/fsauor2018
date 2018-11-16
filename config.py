"""
The config

Note:
 Copy this file to config_local.py and change the path.
 Or, put the data files in directories as showing blow.
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
    # format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')

model_save_path = "data/models/"
train_data_path = "data/train/sentiment_analysis_trainingset.csv"
valid_data_path = "data/valid/sentiment_analysis_validationset.csv"
testa_data_path = "data/test-a/sentiment_analysis_testa.csv"
testa_predict_out_path = "data/testa_prediction.csv"
testb_data_path = "data/test-b/sentiment_analysis_testb.csv"
testb_predict_out_path = "data/testb_prediction.csv"
