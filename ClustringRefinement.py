import glob
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans


# explicit function to normalize array
def normalize(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))

    return x_norm

names = glob.glob('/Path/To/Test/Thumbnails/*.png')
names = [os.path.split(name)[1] for name in names]
# print(names)
# folders = glob.glob('/home/soroush47/fastpathology/projects/VibekesAnnotations/results/*')


for name in names:

    print("/Path/To/images/" + name)
    FM = cv2.imread("/Path/To/Test/PWC/results/" + name)[...,1]/255
    Gr = cv2.imread("/Path/To/Test/Gradients/" + name)[...,1]/255
    Rw1 = cv2.imread("/Path/To/Test/Thumbnails/" + name)[...,0]/255
    Rw2 = cv2.imread("/Path/To/Test/Thumbnails/"  + name)[...,1]/255
    Rw3 = cv2.imread("/Path/To/Test/Thumbnails/"  + name)[...,2]/255
    SP = cv2.imread("/Path/To/Test/Superpixels/" + name)[...,1]/255

    scale_percent = 30 # percent of original size
    width = int(Rw1.shape[1] * scale_percent / 100)
    height = int(Rw1.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    Rw1 = cv2.resize(Rw1, dim, interpolation = cv2.INTER_AREA)
    Rw2 = cv2.resize(Rw2, dim, interpolation = cv2.INTER_AREA)
    Rw3 = cv2.resize(Rw3, dim, interpolation = cv2.INTER_AREA)
    FM = cv2.resize(FM, dim, interpolation = cv2.INTER_AREA)
    Gr = cv2.resize(Gr, dim, interpolation = cv2.INTER_AREA)
    SP = cv2.resize(SP, dim, interpolation = cv2.INTER_AREA)

    FM = normalize(FM)
    FM[FM<0.7] = 0

    Ws = np.array([1, 1, 1, 0.8, 0.2, 0.4])
    features_initial = [FM, Rw1, Rw2, Rw3, Gr, SP]  # Assuming these are your feature arrays

    # Apply the weights to each feature using map
    weighted_features = list(map(lambda f, w: f * w, features, Ws))

    # Stack the weighted features to create a feature vector for each pixel
    features_stacked = np.stack(weighted_features, axis=-1)

    # Stack all images to create a feature vector for each pixel
    features = features_stacked.reshape(-1, 5)

    # Apply KMeans clustering with a consistent initialization and random seed
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(features)

    # Reshape the labels to the image's shape
    labels_2D = labels.reshape(FM.shape)

    # Find the cluster that overlaps the most with the white regions in FM
    overlap_scores = [np.sum((labels_2D == i) & (FM == 1)) for i in range(3)]
    main_cluster = np.argmax(overlap_scores)


    # Replace the main cluster with 1 and other clusters with 0
    pred = np.where(labels_2D == main_cluster, 1, 0)

    label=pred.astype(np.uint8)
    label = cv2.medianBlur(label, 3)

    def fill_holes(binary_img):
        # Copy the image
        im_in = binary_img.copy()

        # Threshold (to ensure binary input)
        th, im_th = cv2.threshold(im_in, 0.45, 1, cv2.THRESH_BINARY_INV)

        # Copy the thresholded image
        im_floodfill = im_th.copy()

        # Mask used for flood filling. Notice the size needs to be 2 pixels larger than the image
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Flood fill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground
        filled_image = im_th | im_floodfill_inv

        return filled_image

    label = fill_holes(label)

    smoothed_image = cv2.blur(label, (79,79))
    smoothed_image = cv2.threshold(smoothed_image,10, 200, cv2.THRESH_BINARY)
    Gr = cv2.imread("/Path/To/Test/Gradients/" + name)[...,1]
    Gr = cv2.resize(Gr, dim, interpolation = cv2.INTER_AREA)
    Gr = cv2.medianBlur(Gr, 11)
    ret,thresh = cv2.threshold(Gr,10,51,cv2.THRESH_BINARY)
    # print(np.unique(thresh))
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    empt = np.zeros(Rw2.shape)
    smoothed_image[1][thresh<0.5]=0
    smoothed_image[1][smoothed_image[1]>100]=255
    Final = cv2.medianBlur(smoothed_image[1], 21)
    cv2.imwrite( '/Path/To/Test/ClusteringResults/' + name, Final)