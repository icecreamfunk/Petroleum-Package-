import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('TM-05.txt')
dataset = dataset.reset_index().values

# dbscan=DBSCAN(eps=3,min_samples=4)
#
# model=dbscan.fit(dataset )
# core_samples_mask = np.zeros_like(dataset.labels_, dtype=bool)
# core_samples_mask[dataset.core_sample_indices_] = True
# labels = dataset.labels_

def dbscan(X, eps, min_samples):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    models = db.fit(X)
    labels = models.labels_
    print(labels)
    y_pred = db.fit_predict(X)
    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=50, color='red')
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=50, color='yellow')
    plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=50, color='cyan')
    plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=50, color='blue')

    plt.title("DBSCAN")

dbscan(dataset,3,4)
# plt.show()