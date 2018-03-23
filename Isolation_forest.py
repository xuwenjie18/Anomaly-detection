# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:50:22 2018

@author: xuwenjie
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
user ='ACM2278' #['ACM2278', 'CMP2946', 'PLJ1771','CDE1846','MBG3183','HIS1706']
data = np.float32(np.load(user+"_data_weekday.npy"))
x_data, y = data[:,1:], data[:,0]
x_std = sc_x.fit_transform(x_data).T

#development set and test set split
#split_index = int(0.8*x_std.shape[1])
#x_std_train, x_std_test  = x_std[:,:split_index], x_std[:,split_index:]

#isolation forest
from sklearn.ensemble import IsolationForest
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples = 'auto', random_state = rng,contamination = 0.001)
clf.fit(x_std.T)  ## n_sample X n_feature
average_depth = clf.decision_function(x_std.T) 
anomly_score3 = np.exp(-10.0*(average_depth-np.max(average_depth))) #convert to anomly score
plt.figure(figsize=(8,8))
a = plt.plot(anomly_score3,'go',alpha=0.3)  
b = plt.plot(np.where(y==1)[0],anomly_score3[np.where(y==1)],'rx')
plt.legend(b,['threat day'],prop=matplotlib.font_manager.FontProperties(size=12),loc='upper left')
plt.title("Isolation Forest")
plt.xlabel("day")
plt.grid(True)