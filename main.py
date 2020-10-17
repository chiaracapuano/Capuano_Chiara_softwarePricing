import pandas as pd

test_results = pd.read_csv("test_results.txt")
test_results.drop(test_results.columns[0], axis=1, inplace=True)
user_table = pd.read_csv("user_table.txt")


df = test_results.merge(user_table, on ='user_id')


#from pandas_profiling import ProfileReport
#profile = ProfileReport(df, title='Pandas Profiling Report')
#profile.to_file(output_file='profile.html')

df.drop(columns = ['lat', 'long', 'user_id', 'country'], inplace=True)
df['operative_system']=df.groupby('device').operative_system.transform(lambda x: x.fillna(x.mode()[0]))
df = df.fillna(df.mode().iloc[0])
print(df['timestamp'][0])

df['timestamp'] =df['timestamp'].str.replace('60', '00')
df['timestamp'] =pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors = 'coerce')
print(df['timestamp'][0].hour)
print(type(df['timestamp'][0]))


df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday
df['year'] = df['timestamp'].dt.year
#check for seasonality
print(df['month'].value_counts())
#check how historical the data is
print(df['year'].value_counts())

df.drop(columns = ['month', 'year', 'timestamp'], inplace=True)

#check trial users were actually displayed 59

print(len(df[df['test']==1]))
print(len(df[df['price']==59.0]))

#remove inconsistent rows
df = df[(df['test']==1) & (df['price']==59.0) | (df['test']==0) & (df['price']==39.0)]

print(len(df))


#calculate the proportional revenue obtained by the test users vs non test users (google significance of ab test)












#users clusters
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_enc = df.apply(le.fit_transform)


#clustering with kmodes

from kmodes.kmodes import KModes

#cost=[]
#for k in list(range(4,11)):
#        print('Iteration number:', k)
#        km = KModes(n_clusters=k, n_init = 1, verbose=1)
#        km.fit(df_enc)
#        cost.append(km.cost_)

#import numpy as np
#y = np.array([i for i in range(4,11,1)])
#import matplotlib.pyplot as plt

#plt.plot(y,cost)
#plt.show()

#choose k=5
km = KModes(n_clusters=5, n_init = 1, verbose=1)
clusters = km.fit_predict(df_enc)

print('done with kmodes')

df_ind_res = df_enc.reset_index()


clustersDf = pd.DataFrame(clusters)
clustersDf.columns = ['cluster_predicted']
df_clusters = pd.concat([clustersDf, df_ind_res], axis = 1).reset_index()
df_clusters = df_clusters.drop(['index', 'level_0'], axis = 1)

print(df_clusters.head())

import matplotlib.pyplot as plt
import seaborn as sns
for col in df_enc.columns:
    plt.subplots(figsize = (15,5))
    sns.countplot(x=df_clusters[col],order=df_clusters[col].value_counts().index,hue=df_clusters['cluster_predicted'])
    plt.show()