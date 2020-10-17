
import pandas as pd

test_results = pd.read_csv("test_results.txt")
test_results.drop(test_results.columns[0], axis=1, inplace=True)
user_table = pd.read_csv("user_table.txt")

df = test_results.merge(user_table, on ='user_id')


df.drop(columns = ['lat', 'long', 'user_id', 'country'], inplace=True)


df['operative_system']=df.groupby('device').operative_system.transform(lambda x: x.fillna(x.mode()[0]))
df = df.fillna(df.mode().iloc[0])


df['timestamp'] =df['timestamp'].str.replace('60', '00')
df['timestamp'] =pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors = 'coerce')


df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday
df['year'] = df['timestamp'].dt.year
df['weeknum'] = df['timestamp'].dt.week


print('Test months:',df['month'].value_counts())
#check how historical the data is
print('Test years:',df['year'].value_counts())




df.drop(columns = ['year', 'timestamp'], inplace=True)



print(len(df[df['test']==1]))
print(len(df[df['price']==59.0]))



df = df[(df['test']==1) & (df['price']==59.0) | (df['test']==0) & (df['price']==39.0)]




df.drop(columns = ['price'], inplace=True)
df = df.dropna()


from scipy.stats import chi2_contingency

data_crosstab = pd.crosstab(df['test'],
                            df['converted'],
                            margins = False,
                            normalize = 'index')
chi2, p, dof, ex= chi2_contingency(data_crosstab)

print(p)


print('min month',df['month'].min())
days = df[df['month']==3.0]
print('min day in month 3',df['weekday'].min())

#%% chi2 test in one week span

min_week_num = df['weeknum'].min()
first_week = df[(df['month']==3.0) & (df['weeknum']==min_week_num)]

#%%

data_crosstab = pd.crosstab(first_week['test'],
                            first_week['converted'],
                            margins = False,
                            normalize = 'index')
chi2, p, dof, ex= chi2_contingency(data_crosstab)

print(p)



df.drop(columns = ['month','weeknum','test'], inplace=True)


from sklearn import preprocessing
df_enc = df
to_encode = ['source', 'device', 'operative_system', 'city']
le = preprocessing.LabelEncoder()

df_enc[to_encode] = df_enc[to_encode].apply(le.fit_transform)





from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score

silhouette=[]
cluster_n=[]
for k in list(range(2,10)):
        print('Iteration number:', k)
        km = KModes(n_clusters=k, n_init = 1, verbose=1)
        preds = km.fit_predict(df_enc)
        centers = km.cluster_centroids_

        score = silhouette_score(df_enc, preds)
        cluster_n.append(k)
        silhouette.append(score)
        print("For n_clusters = {}, silhouette score is {})".format(k, score))

