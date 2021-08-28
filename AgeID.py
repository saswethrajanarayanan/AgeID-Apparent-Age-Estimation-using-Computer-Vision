from keras.models import Model, load_model
import numpy as np
import cv2
import dlib
import tensorflow as tf
from keras.layers import Dense, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Input, Dropout, concatenate, AveragePooling2D
from keras.regularizers import l2
from keras.activations import softmax, relu
#from AgeIDTest import featExtract
from sklearn.cluster import MiniBatchKMeans
import pickle
from collections import deque

des = cv2.ORB_create()
print("OM SAI")
Q = deque(maxlen = 5)

filename = "K_meansModel.sav"
KMeans = pickle.load(open(filename, 'rb'))

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def HCFeat():
  model_in = Input(shape = (100, ))
  hc_model = Dense(256, kernel_regularizer = l2(0.01))(model_in)
  #hc_model = Dense(512, activation='relu')(hc_model)
  #hc_model = Dense(256, activation='relu')(hc_model)
  hc_model = Dropout(0.8)(hc_model)
  #hc_model = Dense(288)

  model = Model(inputs = model_in, outputs = hc_model)
  return model

k = (7, 7)
def build_model():
    i = Input(shape=(64, 64, 3))
    # x = Conv2D((7, 7), 32, activation = 'relu', strides = (1, 1), padding = 'same')(i)
    x = Conv2D(32, k, activation='relu', strides=(2, 2), padding='same')(i)
    p = BatchNormalization()(x)
    x = Conv2D(32, k, activation='relu', strides=(1, 1), padding='same')(p)
    x = BatchNormalization()(x)
    x = Conv2D(32, k, activation='relu', strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, k, activation='relu', strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    semi_merge = concatenate([p, x])
    x = Conv2D(32, k, activation='relu', strides=(1, 1), padding='same')(semi_merge)
    x = BatchNormalization()(x)
    x = Conv2D(32, k, activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(32, k, activation='relu', strides=(1, 1), padding='same')(x)
    x = AveragePooling2D(pool_size=(32, 32))(x)
    x = Flatten()(x)
    model = Model(inputs = i, outputs = x)
    return model

def BuildHybrid():
    CNN = build_model()
    ORB = HCFeat()
    # return CNN, ORB
    merged = concatenate([CNN.output, ORB.output])
    hybrid = Dense(288, activation='relu')(merged)
    hybrid = Dense(101, activation='softmax')(hybrid)
    model = Model(inputs=[CNN.input, ORB.input], outputs=hybrid)
    return model

checkpoint_path = "checkpoint"
#model = build_model()
#model.load_weights(checkpoint_path)
model = BuildHybrid()
model.load_weights("AgeIDorb.h5")
#modelpath = "saved_model.pb"
#model = load_pb(modelpath)
#model = load_model(modelpath)

# #input = input = model.get_tensor_by_name('input:0')
# #output = model.get_tensor_by_name('output:0')
# #sess = tf.Session(graph=model)
#
def clustering(vector, KMeans):
    kp, desc = des.detectAndCompute(vector, None)
    vector_100 = np.zeros(100, dtype='uint8')
    desc = np.array(desc)
    if (len(kp) == 0):
        print("kp invalid")
        return vector_100
    print("kp valid")
    predictions = KMeans.predict(desc)
    #predictions = np.array(predictions)
    #predictions.reshape(1, -1)
    for preds in predictions:
        vector_100[preds] += 1
    return vector_100

cascpath = "C:\\Users\\sasweth\\Downloads\\haarcascade_frontalface_default.xml"

classifier = cv2.CascadeClassifier(cascpath)
cap = cv2.VideoCapture(0)


#labels = np.load("labels.npy")
faces = []
while(True):
    ret, frame = cap.read()
    #print(ret)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    #frame = cv2.resize(frame, (224, 224))
    face = classifier.detectMultiScale(gray, 1.8, 1)
    #print(face)
    if face is ():
        continue
    x, y, w, h = face[0]
    roi = frame[y:y + h, x:x + w]
    # print(len(roi))
    #print(roi.shape)
    faces = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    #print(faces.shape)
    faces = np.array(faces)
    faces = np.expand_dims(faces, axis=0)
    #print(faces.shape)
    xframe = cv2.resize(frame, (64, 64))
    orbfaces = clustering(frame, KMeans = KMeans)
    orbfaces = [orbfaces]
    orbfaces = np.array(orbfaces)
    #face = cv2.resize()
    result = model.predict([faces, orbfaces])
    #print('all labels', result.shape)
    #print('result', max(result[0]))
    pred_age = np.arange(0, 101).reshape(101, 1)
    #print('pred_age', pred_age)
    age = result.dot(pred_age)
    #print(age)
    Q.append(int(age[0][0]))
    result_final = np.array(Q).mean(axis = 0)
    #label = sess.run(output, feed_dict={input: face})
    text = "{}".format(int(result_final))
    print(text)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
    cv2.imshow('Face Detection', frame)

    key = 27
    if cv2.waitKey(100) == 27 & 0xFF:
        break
cv2.destroyAllWindows()






