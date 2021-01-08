#!/usr/bin/env python
# coding: utf-8

# In[3]:



import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from IPython.display import display
import glob
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils



# In[4]:



my_model = load_model('model_cnn.h5')
prototxt_path = "deploy.prototxt"
weight_path = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_path, weight_path)


# In[5]:


def detectPredictMask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Compute the (x, y)-coordinates of the bounding box for the object
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # Ensure the bounding boxes fall within the dimensions of the frame
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    # Extract the face ROI, convert it from BGR to RGB channel
    # Ordering, resize it to 256x256, and preprocess it
    face = frame[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    img = np.array(face, dtype='float')
    img = img.reshape(1, 224, 224, 3)
    
    # Predict the frame
    preds = maskNet.predict(img)

    # Return a 2-tuple of the face locations and their corresponding locations
    return ((startX, startY, endX, endY), preds[0][0])


# In[ ]:


# for fn in uploaded.keys():
#     path = fn
#     img = image.load_img(pasth,target_size=(224,224))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
    
#     img=np.array(img, dtype = 'float')
#     img = img.reshape (1,224,224,3)
#     model = load_model()
#     my_model = load_model ('model.h5')
#     prediksi= model.predict(img)
#     idx = prediksi[0][0]
#     if (idx):
#         print("Not Wearing Masker")
#     else:
#         print("wearing masker")


# In[ ]:


vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame)
    
    (locs, preds) = detectPredictMask(frame, faceNet, my_model)
    
    # Unpack the bounding box and predictions
    (startX, startY, endX, endY) = locs
    result = preds
        
    # Determine the class label and color we'll use to draw the bounding box and text
    color = (0, 225, 0)
    status ="Wearing Mask"
    if (result == 1):
        status ="not Wearing Mask"
        color = (0, 0, 225)

    font = cv2.FONT_HERSHEY_DUPLEX

    stroke = 1
    cv2.putText(frame, status, (startX, startY - 10), font, 0.5, color, stroke, cv2.LINE_AA)

    stroke = 2
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, stroke)
        
    # Showing frame
    cv2.imshow('DETECTING', frame)
    
    # Turn of camera 
    if cv2.waitKey(2) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        vs.stop()

