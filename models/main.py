import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def predict_color(image, no_of_clusters=5):
    scaler = StandardScaler(with_mean=0, with_std=1)

    r = []
    g = []
    b = []

    for line in image:
        for pixel in line:
            temp_b, temp_g, temp_r = pixel

            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)

    df = pd.DataFrame({'red': r, 'green': g, 'blue': b})

    df['scaled_red'] = scaler.fit_transform(df[['red']])
    df['scaled_green'] = scaler.fit_transform(df[['green']])
    df['scaled_blue'] = scaler.fit_transform(df[['blue']])

    X = df[['scaled_red', 'scaled_green', 'scaled_blue']].values

    for cluster in range(2, 8):
        kmeans = KMeans(n_clusters=cluster, random_state=42)
        kmeans.fit(X)

        kmeans.predict(X)

    kmeans = KMeans(n_clusters=no_of_clusters)
    kmeans.fit(X)
    kmeans.predict(X)

    # These are the centroids of the clusters
    cluster_centers = kmeans.cluster_centers_

    colors = []

    r_std, g_std, b_std = df[['red', 'green', 'blue']].std()

    for cluster_center in cluster_centers:
        scaled_r, scaled_g, scaled_b = cluster_center

        colors.append((
            scaled_r * r_std / 255,
            scaled_g * g_std / 255,
            scaled_b * b_std / 255
        ))

    plt.imshow([colors])
    plt.savefig('output.jpg')



