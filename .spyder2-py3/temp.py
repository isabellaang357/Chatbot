from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from subprocess import Popen


np.set_printoptions(threshold=np.nan)
df=pd.read_excel('lenses.xlsx', header= None)
p=Popen('lenses.xlsx', shell=True)
A=df.as_matrix()
train_x=A[0:250,0:6]
train_y=A[0:250,6]

reg=MLPRegressor(hidden_layer_sizes=(50,),max_iter=2000,solver='lbfgs')
reg.fit(train_x,train_y)

output=reg.predict(train_x)
plt.plot(train_y)
plt.plot(output)
plt.show()