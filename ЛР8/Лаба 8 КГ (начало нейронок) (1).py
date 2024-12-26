#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from mnist import MNIST

img_rows = img_cols = 28
path_to_data = (r'C:\Users\cjabz\Downloads\mnist_bin')

def load_data(path_to_data, img_rows, img_cols):
    print('Загрузка данных из двоичных файлов...')
    with open(path_to_data + '\images_trn.bin', 'rb') as rb:
        x_trn = np.fromfile(rb, dtype = np.uint8)
    with open(path_to_data + '\labels_trn.bin', 'rb') as rb:
        y_trn = np.fromfile(rb, dtype = np.uint8)
    with open(path_to_data + '\images_tst.bin', 'rb') as rb:
        x_tst = np.fromfile(rb, dtype = np.uint8)
    with open(path_to_data + '\labels_tst.bin', 'rb') as rb:
        y_tst = np.fromfile(rb, dtype = np.uint8)
    x_trn = x_trn.reshape(-1, img_rows * img_cols)
    x_tst = x_tst.reshape(-1, img_rows * img_cols)
    return x_trn, y_trn, x_tst, y_tst
# Загрузка обучающего и проверочного множества из бинарных файлов
# Загружаются изображения и их метки
x_trn, y_trn, x_tst, y_tst = load_data(path_to_data, img_rows, img_cols)
print("End")

# def plot_images(images, labels, num_images=10):
#     plt.figure(figsize=(10, 2))  # Размер фигуры увеличен для более удобного отображения
#     plt.imshow(images[i].reshape(img_rows, img_cols), cmap='gray')
#     plt.title(labels[i])
#     plt.axis('off')  
#     plt.show()

# # Вывод первых 1000 изображений из обучающего набора
# for i in range (10):
#     plot_images(x_trn, y_trn)


# In[15]:


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

linear_regressor.fit(x_trn, y_trn)


# In[16]:


from sklearn.metrics import classification_report
y_true = y_tst
y_pred = linear_regressor.predict(x_tst).astype('int32')

unique_classes = np.unique(y_tst)

print(y_pred)
target_names = [f'class {i}' for i in unique_classes]

# target_names[0] = 'Кластер 0 -- ' + target_names[0]

print(classification_report(y_tst, y_pred, target_names=target_names, labels=unique_classes))


# In[ ]:


# # Визуализация центроидов
# def plot_centroids(centroids, labels):
#     plt.figure(figsize=(10, 2))
#     plt.imshow(centroids.reshape(img_rows, img_cols), cmap='gray')
#     plt.title(labels)
#     plt.axis('off')
#     plt.show()
    
# for i in range(centroids.shape[0]):
#     plot_centroids(centroids[i], label_map[i])


# In[67]:


from sklearn.metrics import classification_report
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x_trn_norm = x_trn / 255.0  # Приведение пиксельных значений в диапазон [0,1]

# Применим K-Means
k = 10  # Вы предполагаете, что у вас 10 кластеров (соответствуют 10 цифрам MNIST)
kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
kmeans.fit(x_trn_norm)
y_pred = kmeans.predict(x_trn_norm)

# Получить метки кластеров и центроиды
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

cluster_labels = {}
for cluster in range(k):
    # Для каждой метки кластера находите цифру, которая чаще всего присваивается этому кластеру
    mask = (labels == cluster)
    most_common_label = mode(y_trn[mask]).mode
    cluster_labels[cluster] = most_common_label
    
# Выводим карту сопоставления кластеров к цифрам
print("Кластер -> Цифра")
for cluster, digit in label_map.items(): 
    print(f"    {cluster}   ->   {digit}")
    
# Преобразуем предсказанные кластеры в метки
y_pred_labels = np.array([cluster_labels[cluster] for cluster in y_pred])

print(classification_report(y_trn, y_pred_labels, target_names=target_names, zero_division = 0 ))


# In[66]:


x_tst_norm = x_tst / 255.0

y_pred_tst = kmeans.predict(x_tst_norm)
# Получить метки кластеров и центроиды
labels = kmeans.labels_

centroids = kmeans.cluster_centers_

cluster_labels = {}
for cluster in range(k):
    # Для каждой метки кластера находите цифру, которая чаще всего присваивается этому кластеру
    mask = (labels == cluster)
    most_common_label = mode(y_trn[mask]).mode
    cluster_labels[cluster] = most_common_label
    
# Выводим карту сопоставления кластеров к цифрам
print("Кластер -> Цифра")
for cluster, digit in label_map.items(): 
    print(f"    {cluster}   ->   {digit}")
    
# Преобразуем предсказанные кластеры в метки
y_pred_labels = np.array([cluster_labels[cluster] for cluster in y_pred_tst])

print(classification_report(y_tst, y_pred_labels, target_names=target_names, zero_division = 0 ))


# In[ ]:





# In[ ]:




