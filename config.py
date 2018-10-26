"""
The config

Note:
 Copy this file to config_local.py and change the path.
 Or, put the data files in directories as showing blow.
"""
import logging

logging.getLogger('jieba').setLevel(logging.WARN)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
    # format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')

model_save_path = "data/models/"
train_data_path = "data/train/sentiment_analysis_trainingset.csv"
validate_data_path = "data/validate/sentiment_analysis_validationset.csv"
test_data_path = "data/test-a/sentiment_analysis_testa.csv"
test_data_predict_out_path = "data/test_prediction.csv"
