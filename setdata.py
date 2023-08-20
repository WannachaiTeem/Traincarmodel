import requests
import cv2
import base64
import os
import numpy as np
import pickle

url = 'http://localhost:8080/api/gethog/'
# Method เรียกใช้ api แปลงรูปภาพ เพื่อถึงเอา feature vector ของรูปรถ
def featureVector(img_path):
    img = cv2.imread(img_path)
    retval, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer)
    img_base64 = "image data," + str.split(str(img_base64), "'")[1]
    data = {"img": img_base64}
    response = requests.post(url, json=data)
    return response.json()

#Method set Train file and Test File เพื่อ เตรียมข้อมูลสำหรับการสร้าง file train/test dataset
def setTrainTestFile(dataPath):

    x = []
    y = []
    sum_xy = []
    Brands_List = os.listdir(dataPath)

    for brand in Brands_List: 
        brand_path = os.path.join(dataPath, brand)
        cars_List = os.listdir(brand_path)

        for car in cars_List:
            img_file_name = os.path.join(brand_path, car)
            x.append(img_file_name)
            y.append(brand)

    # เรียกใช้ Method featureVector แล้วเก็บค่า hog ไว้ใน lsit sum_xy
    for index, value in enumerate(x):
        hog = featureVector(x[index])
        if hog:
            hog = list(hog['HOG'])
            hog.append(y[index])
            sum_xy.append(hog)
        else:
            print('!!!! HOG is Error !!!!')
    return sum_xy
 
#ทำการบันทึก Feature vector และ Brand(อยู่ index ที่ ตำแหน่งสุดท้ายของ ค่า hog) สำหรับ train ลงไฟในไฟล์นามสกุล .pkl
train_path = r'dataset\train'
Traindataset_FeatureVector = setTrainTestFile(train_path)
trainfile_name = 'TrainFeatureVector.pkl'
pickle.dump(Traindataset_FeatureVector, open(trainfile_name, 'wb'))
print('Finished creating a file called TrainFeatureVector.pkl')


# dataser path ของชุดข้อมูล
test_path = r'dataset\test'
Test_dataset_FeatureVector = setTrainTestFile(test_path)
testfile_name = 'TestFeatureVector.pkl'
pickle.dump(Test_dataset_FeatureVector, open(testfile_name, 'wb'))
print('Finished creating a file called TestFeatureVector.pkl')