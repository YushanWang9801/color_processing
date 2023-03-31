from matplotlib import image as img
import matplotlib.pyplot as plt
import pandas as pd

image = img.imread('./images/1.jpg')
df = pd.DataFrame()
df['r']=pd.Series(image[:,:,0].flatten())
df['g']=pd.Series(image[:,:,1].flatten())
df['b']=pd.Series(image[:,:,2].flatten())

from scipy.cluster.vq import whiten
df['r_whiten'] = whiten(df['r'])
df['g_whiten'] = whiten(df['g'])
df['b_whiten'] = whiten(df['b'])

from scipy.cluster.vq import kmeans
cluster_centers, distortion = kmeans(df[['r_whiten', 'g_whiten', 'b_whiten']], 5)

r_std, g_std, b_std = df[['r', 'g', 'b']].std()
colors=[]
for color in cluster_centers:
    sr, sg, sb = color
    colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
plt.imshow([colors])
plt.show()


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from matplotlib import image as img

image = img.imread('./images/1.jpg')
pixels = image.reshape(-1, 3)
# Perform k-means clustering to extract the dominant colors
kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
centers = kmeans.cluster_centers_

colors = []
for color in centers:
    colors.append((int(color[0]), int(color[1]), int(color[2])))

plt.imshow([colors])
plt.show()