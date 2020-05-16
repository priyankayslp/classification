import itertools
import numpy as np
import pandas as pd
import matplotlib.ticker as tk
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib.ticker import NullFormatter

df = pd.read_csv("data/teleCust1000t.csv")
print(df.head())

print(df['custcat'].value_counts())
print(df['income'].describe())
df.hist(column='income',bins=50)
plt.show()
print(df.columns)
x = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
print(x[0:10])

y = df['custcat'].values
print(y[0:5])
print(df.isnull().sum())
x= preprocessing.StandardScaler().fit(x).transform(x.astype(float))
print(x[0:5])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print("TRAIN set :",x_train.shape,y_train.shape)

from sklearn.neighbors import KNeighborsClassifier
k=4
neigh = KNeighborsClassifier(k).fit(x_train,y_train)
y_hat = neigh.predict(x_test)
print(y_hat[0:5])

from sklearn import metrics
#accuracy score is jaccad index
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_hat))

ks = 32
jacc_index= np.zeros((ks-1))
stan_error = np.zeros((ks-1))
bool_arr=[]
for n in range(1,ks):
    neigh= KNeighborsClassifier(n).fit(x_train,y_train)
    y_hat=neigh.predict(x_test)
    jacc_index[n - 1] = metrics.accuracy_score(y_test, y_hat)
    stan_error[n-1]=np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])

print(jacc_index)

plt.plot(range(1,ks),jacc_index,'g')
plt.fill_between(range(1,ks),jacc_index - 1 * stan_error,jacc_index + 1 * stan_error, alpha=0.1)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", jacc_index.max(), "with k=", jacc_index.argmax()+1)


