import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_breast_cancer
cancer_dataset=load_breast_cancer()
cancer_dataset
print(cancer_dataset.DESCR)
df=pd.DataFrame(cancer_dataset['data'],columns=
                cancer_dataset['feature_names'])
df
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
scaled_data
#applying pca
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
data_pca=pca.fit_transform(scaled_data)
data_pca
pca.explained_variance_#this varience is captured

plt.figure(figsize=(8,6))
plt.scatter(data_pca[:,0],data_pca[:,1],
            c=cancer_dataset['target'],cmap='plasma')
plt.xlabel('first principal component')
plt.ylabel('second principal component')
#now we apply classification problem
