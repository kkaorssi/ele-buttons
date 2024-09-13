import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd

# 랜덤값 고정
# np.random.seed(8)
# X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.9)
# plt.scatter(X[:, 0], X[:, 1], marker='.')
# print(X, type(X), X.shape)
df1 = pd.read_excel('./final/pos_data.xlsx', sheet_name=1, engine='openpyxl')
X = np.array(df1[['X', 'Y']])
print(X)


k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

k_means_labels = k_means.labels_
# print('k_means_labels : ', k_means_labels)

k_means_cluster_centers = k_means.cluster_centers_
print('k_means_cluster_centers : ', k_means_cluster_centers)

# 지정된 크기로 초기화
fig = plt.figure(figsize=(6, 4))

# 레이블 수에 따라 색상 배열 생성, 고유한 색상을 얻기 위해 set(k_means_labels) 설정
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# plot 생성
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(4), colors):
    my_members = (k_means_labels == k)

    # 중심 정의
    cluster_center = k_means_cluster_centers[k]

    # 중심 그리기
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('K-Means')
ax.set_xticks(())
ax.set_yticks(())
plt.show()