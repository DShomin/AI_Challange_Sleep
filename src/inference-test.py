#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

os.system("python inference.py -C config_b0_0")
os.system("python inference.py -C config_b0_1")
os.system("python inference.py -C config_b1")
os.system("python inference.py -C config_b2")
os.system("python inference.py -C config_b3")
os.system("python inference.py -C config_b4")


b00 = np.load('tf_efficientnet_b0_ns_0.8/test_preds.npy') 
b01 = np.load('tf_efficientnet_b0_ns_0.7/test_preds.npy') 
b1 = np.load('tf_efficientnet_b1_ns_0.8/test_preds.npy')    
b2 = np.load('tf_efficientnet_b2_ns_0.8/test_preds.npy')   
b3 = np.load('tf_efficientnet_b3_ns_0.8/test_preds.npy')   
b4 = np.load('tf_efficientnet_b4_ns_0.8/test_preds.npy')  



test_preds = np.argmax(((b00+b01)/2 + b1 + b2 + b3 + b4)/5, 1)

result_df = pd.DataFrame(test_preds)
label_dict = {0:'Wake', 1:'N1', 2:'N2', 3:'N3', 4:'REM'}
result_df[0] = result_df[0].map(label_dict)
print(result_df.loc[:10], result_df.shape)
test_pred_path = "/USER/INFERENCE"
result_df.to_csv(os.path.join(test_pred_path, 'final_result_ensemble.csv'), header=None, index=False)
