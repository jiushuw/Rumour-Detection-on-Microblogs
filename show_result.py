import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 定义两种颜色
cmap = ListedColormap(['blue', 'red'])

X = np.load('fusion_features.npy')
y = np.load('label_fusion_features.npy')
print(X.shape)
print(X.dtype)
print(y.shape)
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap)
plt.title('t-SNE')
plt.show()
