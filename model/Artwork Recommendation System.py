#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os, os.path
import random
import re
from tqdm.auto import tqdm
import cv2
import glob
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tensorflow.keras import applications
from k_means_constrained import KMeansConstrained
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pandas as pd
import numpy as np


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ### Dataset Information

# In[ ]:


# directory where images are stored
DIR = "/home/dxlab/jupyter/eunhoo/wikiart"
folders = [os.path.basename(x) for x  in os.listdir(DIR)]


# In[ ]:


stats = []
for folder in folders:
        # get a list of subdirectories that start with this character
        directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/*',recursive=True)]
        
        for sub_directory in directory_list:
            file_names = [file for file in os.listdir('/home/dxlab/jupyter/eunhoo/wikiart/'+ sub_directory)]
            file_count = len(file_names)
            sub_directory_name = os.path.basename(sub_directory)
            stats.append({ "Image count": file_count, 
                           "Folder name": os.path.basename(sub_directory),
                            "File names": file_names})
            
        df = pd.DataFrame(stats)


# In[ ]:


# 사용할 style만 가져오기
use_folders = ['Action_painting', 'Baroque', 'Impressionism', 'Pop_Art', 'Cubism',
       'Minimalism', 'Color_Field_Painting', 'New_Realism', 'Ukiyo_e',
       'High_Renaissance']
use_df = df.loc[df['Folder name'].isin(use_folders),:]


# #### Concat Folders

# In[ ]:


# Action painting
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Action_painting/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Action_painting',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset1 = pd.DataFrame(df)
    dataset1 = dataset1.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Baroque
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Baroque/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Baroque',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset2 = pd.DataFrame(df)
    dataset2 = dataset2.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Impressionism
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Impressionism/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Impressionism',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset3 = pd.DataFrame(df)
    dataset3 = dataset3.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Pop Art
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Pop_Art/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Pop_Art',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset4 = pd.DataFrame(df)
    dataset4 = dataset4.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Cubism
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Cubism/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Cubism',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset5 = pd.DataFrame(df)
    dataset5 = dataset5.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Minimalism
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Minimalism/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Minimalism',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset6 = pd.DataFrame(df)
    dataset6 = dataset6.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Color Field Painting
directory_list = [os.path.basename(x) for x in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Color_Field_Painting/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Color_Field_Painting',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset7 = pd.DataFrame(df)
    dataset7 = dataset7.sort_values('filename').reset_index(drop=True)


# In[ ]:


# New_Realism
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/New_Realism/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'New_Realism',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset8 = pd.DataFrame(df)
    dataset8 = dataset8.sort_values('filename').reset_index(drop=True)


# In[ ]:


# Ukiyo_e
directory_list = [os.path.basename(x) for x  in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/Ukiyo_e/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'Ukiyo_e',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset9 = pd.DataFrame(df)
    dataset9 = dataset9.sort_values('filename').reset_index(drop=True)


# In[ ]:


# High Renaissance
directory_list = [os.path.basename(x) for x in glob.glob('/home/dxlab/jupyter/eunhoo/wikiart/High_Renaissance/*',recursive=True)]

df = []
for image in directory_list:
    artist = image.split('_')[0]
    artwork = image.split('_')[1]
    filename = image
    df.append({'genre':'High_Renaissance',
               'artist':artist,
              'artwork':artwork,
              'filename':image})
    
    dataset10 = pd.DataFrame(df)
    dataset10 = dataset10.sort_values('filename').reset_index(drop=True)


# In[ ]:


dataset = pd.concat([dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,
                    dataset7,dataset8,dataset9,dataset10])


# In[ ]:


dataset.drop_duplicates(subset = ['genre','filename'], inplace=True)
dataset.reset_index(drop = True, inplace=True)


# In[ ]:


dataset = pd.read_csv("total_dataset.csv")
len(dataset)


# ### Normalize & Load Images

# In[ ]:


# to get the files in alphabetic order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

# load images selected genres
def load_images(genre):
    
    images = []
    labels = []
    

    path = DIR + '/' + genre
    files = os.listdir(path)
    files = sorted_alphanumeric(files)

    for file in tqdm(files):
        image = cv2.imread(path + '/' + file,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        image = image.astype('float32') /255.0
        images.append(img_to_array(image))
        labels.append(genre)
        
    return images,labels


# In[ ]:


def normalize_images(images, labels):

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Normalise the images
    images /= 255.0
    
    return images, labels


# In[ ]:


def covnet_transform(covnet_model, raw_images):

    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat


# In[ ]:


# imagenet에 미리 훈련된 ResNet50 모델 불러오기
model = ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
model.summary()
 
X_result = None


for idx, g in enumerate(use_folders) :
    print("========", g, "========")
    
    images, labels = load_images(g)
    images, labels = normalize_images(images, labels)
    
    if idx == 0 :
        y = labels
        result = covnet_transform(model, images)
    else :
        y = np.concatenate((y, labels))
        result = np.concatenate((result, covnet_transform(model, images)))


# ### PCA

# In[ ]:


pca = PCA(n_components=2, random_state=10)
resnet_output_pca = pca.fit_transform(result)
principalDF = pd.DataFrame(data=resnet_output_pca,columns = ['principal component1','principal component2'])


# In[ ]:


y_categorical = [use_folders.index(x) for x in y]


# ### Clustering
# 
# #### 1) k-means 

# In[ ]:


kmeans  = KMeans(n_clusters=7,random_state=12)
kmeans.fit(resnet_output_pca)     


# In[ ]:


from collections import Counter
Counter(kmeans.labels_)


# In[ ]:


from sklearn.metrics import pairwise_distances

distances = pairwise_distances(kmeans.cluster_centers_, np.array(principalDF), metric='euclidean')
ind = [np.argpartition(i,10)[:10] for i in distances]
closest = [np.array(principalDF)[indexes] for indexes in ind]


# In[ ]:


cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_


# In[ ]:


dataset20 = pd.concat([dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,
                    dataset7,dataset8,dataset9,dataset10])
dataset20.drop_duplicates(subset = ['genre','filename'], inplace=True)
dataset20.reset_index(drop = True, inplace=True)


# In[ ]:


image_cluster = pd.DataFrame(dataset20['filename'].tolist(),columns=['image'])
image_cluster['genre'] = dataset20['genre'].tolist()
image_cluster['clusterid'] = labels


# #### Top 10 closest images from each cluster center

# In[ ]:


cluster1_ind = ind[0]
cluster2_ind = ind[1]
cluster3_ind = ind[2]
cluster4_ind = ind[3]
cluster5_ind = ind[4]


# In[ ]:


# cluster 1
for i in cluster1_ind:
    fig = plt.figure(figsize=(10,7))
    rows = 2
    columns = 5
    fig.add_subplot(rows, columns, 1)
    
    cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    cluster_image_close = cv2.imread(cluster_path)
    cluster_image_close = cv2.cvtColor(cluster_image_close,cv2.COLOR_BGR2RGB)
    cluster_image_close = cv2.resize(cluster_image_close,(224,224))
    plt.imshow(cluster_image_close)
    plt.show()


# In[ ]:


# cluster 2
for i in cluster2_ind:
    fig = plt.figure(figsize=(10,7))
    rows = 2
    columns = 5
    fig.add_subplot(rows, columns, 1)
    
    cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    cluster_image_close = cv2.imread(cluster_path)
    cluster_image_close = cv2.cvtColor(cluster_image_close,cv2.COLOR_BGR2RGB)
    cluster_image_close = cv2.resize(cluster_image_close,(224,224))
    plt.imshow(cluster_image_close)
    plt.show()


# In[ ]:


# cluster 3
for i in cluster3_ind:
    fig = plt.figure(figsize=(10,7))
    rows = 2
    columns = 5
    fig.add_subplot(rows, columns, 1)
    
    cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    cluster_image_close = cv2.imread(cluster_path)
    cluster_image_close = cv2.cvtColor(cluster_image_close,cv2.COLOR_BGR2RGB)
    cluster_image_close = cv2.resize(cluster_image_close,(224,224))
    plt.imshow(cluster_image_close)
    plt.show()


# In[ ]:


# cluster 4
for i in cluster4_ind:
    fig = plt.figure(figsize=(10,7))
    rows = 2
    columns = 5
    fig.add_subplot(rows, columns, 1)
    
    cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    cluster_image_close = cv2.imread(cluster_path)
    cluster_image_close = cv2.cvtColor(cluster_image_close,cv2.COLOR_BGR2RGB)
    cluster_image_close = cv2.resize(cluster_image_close,(224,224))
    plt.imshow(cluster_image_close)
    plt.show()


# In[ ]:


# cluster 5
for i in cluster5_ind:
    fig = plt.figure(figsize=(10,7))
    rows = 2
    columns = 5
    fig.add_subplot(rows, columns, 1)
    
    cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    cluster_image_close = cv2.imread(cluster_path)
    cluster_image_close = cv2.cvtColor(cluster_image_close,cv2.COLOR_BGR2RGB)
    cluster_image_close = cv2.resize(cluster_image_close,(224,224))
    plt.imshow(cluster_image_close)
    plt.show()


# In[ ]:


labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


# ### Recommendation
# #### 1) extract liked pictures

# In[ ]:


# like 한 그림들을 random으로 추출(cluster의 중심의 그림들이 각 cluster의 대표 이미지라는 가정)
choice_list = [random.sample(cluster_centers.tolist(),5)]


# In[ ]:


dict_cluster_centers = {idx : pts for idx, pts in enumerate(cluster_centers)}


# In[ ]:


choice_idx = random.sample(dict_cluster_centers.keys(), 5)
choice_list_jy = {idx : dict_cluster_centers[idx].tolist() for idx in choice_idx}
choice_list_jy


# In[ ]:


# liked한 대표 그림들의 중심 위치와 중심이 속하는 cluster 찾기
final_center_jy = np.mean(list(choice_list_jy.values()), axis = 0)
final_center_jy


# #### 2) Extract random pictures from closest center cluster

# In[ ]:


# find the cluster in which the center is included
dist = list()
for i in dict_cluster_centers:
    dist.append(np.linalg.norm(final_center_jy-dict_cluster_centers[i]))
    closest_cluster = dist.index(min(dist))
print(closest_cluster)


# In[ ]:


close_idx = [idx for idx, label in enumerate(labels) if label == closest_cluster]


# In[ ]:


# extract random pictures from the selected cluster
min_choice_list = random.sample(close_idx,10)
min_choice_list


# In[ ]:


for i in min_choice_list:
    min_cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    min_cluster_image_close = cv2.imread(min_cluster_path)
    min_cluster_image_close = cv2.cvtColor(min_cluster_image_close,cv2.COLOR_BGR2RGB)
    min_cluster_image_close = cv2.resize(min_cluster_image_close,(224,224))
    plt.imshow(min_cluster_image_close)
    plt.show() 


# #### 3) Figure out the farthest cluster from the center

# In[ ]:


dist = list()
for i in dict_cluster_centers:
    dist.append(np.linalg.norm(final_center_jy-dict_cluster_centers[i]))
    farthest_cluster = dist.index(max(dist))
print(farthest_cluster)


# In[ ]:


for idx, img_pts in enumerate(resnet_output_pca) :
    print(img_pts)
    break


# In[ ]:


far_idx = [idx for idx, label in enumerate(labels) if label ==farthest_cluster]


# In[ ]:


max_choice_list = random.sample(far_idx,10)

for i in max_choice_list:
    max_cluster_path = DIR + '/' + dataset20['genre'][i] + '/' + image_cluster['image'][i]
    max_cluster_image_close = cv2.imread(max_cluster_path)
    max_cluster_image_close = cv2.cvtColor(max_cluster_image_close,cv2.COLOR_BGR2RGB)
    max_cluster_image_close = cv2.resize(max_cluster_image_close,(224,224))
    plt.imshow(max_cluster_image_close)
    plt.show() 

