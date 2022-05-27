'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
import glob
from sklearn.cluster import KMeans
import pandas as pd



import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def load_images_from_folder(folder):
    images = []
    fname=[]
    for filename in os.listdir(folder):
        fname.append(filename)
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images, fname




def detect_faces(input_path: str) -> dict:
    result_list = []
    count=[]
    images, fname=load_images_from_folder(input_path)
    print(fname)
    print(len(fname))
    number_of_images=len(images)
    face = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')
    print(face)
    for i in range(number_of_images):
        img = images[i]
        faces = face.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

        for x,y,w,h in faces:
            face_boundary= {"iname": fname[i], "bbox": [int(x), int(y), int(w), int(h)]}
            result_list.append(face_boundary)
            cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0),2)
    return result_list


''''

    harr_file_path=cv2.data.haarcascades+'haarcascade_frontalface_alt.xml'
    harr_file_path=harr_file_path.replace('/','//')
    face_cascade = cv2.CascadeClassifier(harr_file_path)

    images_path = glob.glob(input_path+ "/*")
    for each_image_path in images_path :
        img_name= "{}".format(os.path.split(each_image_path)[-1].split('.')[0])
        img = cv2.imread(each_image_path, cv2.IMREAD_GRAYSCALE)
        detected_face = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4, minSize=(20, 20),flags=cv2.CASCADE_SCALE_IMAGE)
        draw_rectangle(detected_face,img,img_name,result_list)
    return result_list
'''
'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    detected_faces=detect_faces(input_path)
    data_cluster=[]
    images_name=[]
    for each_value in detected_faces:
        each_image_path=input_path+"/"+each_value['iname']
        img = cv2.imread(each_image_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left=each_value['bbox'][0]
        top=each_value['bbox'][1]
        right=left+each_value['bbox'][2]
        bottom=top+each_value['bbox'][3]
        box_value=(top,right, bottom,left)
        img_array=face_recognition.face_encodings(rgb, [box_value])
        data_cluster.append(list(img_array[0])) 
        images_name.append(each_value['iname'])
       
    kmeans = KMeans(int(K))
    kmeans.fit(data_cluster)
    identified_clusters = kmeans.fit_predict(data_cluster)
    for i in range(5):
        result_list.append({'cluster_no':i, 'elements':[]})
    for cluster,image in zip(identified_clusters,images_name):
        for item in result_list:
            if cluster==item['cluster_no']:
                item['elements'].append(image)
                break

    for each_item in result_list:
        hc=[]
        clust=  each_item['cluster_no']     
        for img_name in each_item['elements']:
           
            img = cv2.imread(input_path+'/'+img_name)  
            img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_NEAREST)
            
            hc.append(img)
        h_img = cv2.hconcat(hc)
        cv2.imwrite('cluster_'+str(clust)+'.jpg', h_img)

    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

def draw_rectangle(detected_face,img,img_name,result_list):
    for (column, row, width, height) in detected_face:
        cv2.rectangle(img,(column, row),(column + width, row + height),(0, 255, 0),2)
        face_boundary= {"iname": img_name+".jpg", "bbox": [int(column), int(row), int(width), int(height)]}
        result_list.append(face_boundary)
