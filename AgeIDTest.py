import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from imutils import paths
from pathlib import Path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
#from mtcnn.mtcnn import MTCNN
import pickle

#detector = MTCNN()
des = cv2.ORB_create()

final_train_ImageDescs = []


defaulters = np.load('defaulters.npy')
def featExtract(image, descriptor):
    #kp = descriptor.detect(image, None)
    kp, desc = descriptor.detectAndCompute(image, None)
    return desc  #return this to final_valid_ImageDescs


#extracted_final below, is the feature descriptors of all the Images in the training set.
def VocabularizeAndBeyond(extracted_final, KMeans_model, n_clusters, final_valid_ImageDescs):
    KMeans_model.fit(extracted_final)   # Be sure to pass a numpy array in here
    clustered_words = [KMeans_model.predict(single_image_features) for single_image_features in final_valid_ImageDescs]
    BOVWHists = np.array([np.bincount(cluster_word, minlength=n_clusters) for cluster_word in clustered_words])
    filename = "K_meansModel.sav"
    pickle.dump(KMeans_model, open(filename, 'wb'))
    return BOVWHists

no_face = []
cascpath = "C:\\Users\\sasweth\\Downloads\\haarcascade_frontalface_default.xml"
facedet = cv2.CascadeClassifier(cascpath)
key = []
appa_dir = "C:\\Users\\sasweth\\Downloads\\appa-real-release"
pathNew = Path(appa_dir)
train_path = pathNew.joinpath("train")
imagePaths = sorted(list(paths.list_images(train_path)))
print("Feature Extraction Pipeline Begins")
count = 0
for impaths in imagePaths:
    if impaths.endswith("_face.jpg"):
        print("count", count)
        image = cv2.imread(impaths)
        if count in defaulters:
            count += 1
            continue
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        descs = featExtract(image, des)
        if descs is None:
            key.append(count)
            count += 1
            continue
        final_train_ImageDescs.append(descs)
        count += 1

np.save('final_train_imagedescsNEW.npy', final_train_ImageDescs)
print(len(final_valid_ImageDescs))
print("delete", key[0:])
print("delete ", len(key))


#finalImageDescs = np.load('final_train_imageDescsNEW.npy', allow_pickle=True)
# print(len(finalImageDescs))

print("Feature Extraction Pipeline Ends")
print("Clustering Begins")
extracted_final = [feature for feature_list in finalImageDescs for feature in feature_list]
extracted_final = np.array(extracted_final)
np.save("extracted_final.npy", extracted_final)
KMeans_model = MiniBatchKMeans(n_clusters= 100)
n_clusters = 100
print("Vocabularizing the feature set")
X_trainNEW = VocabularizeAndBeyond(extracted_final, KMeans_model, n_clusters, finalImageDescs)
np.save("X_trainNEW.npy", X_trainNEW)
print("F-E Routine for 3553 sized valid Data Complete")

# filename = "K_meansModel.sav"
# KMeans = pickle.load(open(filename, 'rb'))
# X = np.load('extracted_final.npy')
# labels = KMeans.labels_

print(metrics.silhouette_score(extracted_final, labels, metric = 'euclidean'))



#Final Testing
#X_valid = np.load('X_validNEW.npy')
#print(X_valid.shape)
#
#
#










