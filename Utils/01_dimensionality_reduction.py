import numpy as np

from sklearn.manifold import TSNE

np_array = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0],
    [1.1, 1.2, 1.3, 1.4, 1.5]
])

"""
创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化.

Args:
    n_components: 降维后的维度（在这里是2D）
    perplexity: 可以理解为近邻的数量
    random_state: 随机数生成器的种子
    init: 初始化方式
    learning_rate: 学习率

Returns:
    The TSNE object.
"""
tsne = TSNE(n_components=2, perplexity=2, random_state=2, init='random', learning_rate=200)

"""
Returns:
    [[-1358.8857    560.47076]
     [ 1937.2167    906.9153 ]
     [  601.5877  -2103.3655 ]]
"""
tsne_dims = tsne.fit_transform(np_array)
