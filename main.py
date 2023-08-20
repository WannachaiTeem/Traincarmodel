import pickle
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#อ่านไฟล์ dataset ที่เก็บ feature vector และ ยี่ห้อรถ(ในตำแหน่ง 8100-1) ของรูปภาพ
train_path = r"TrainFeatureVector.pkl"
test_path = r"TestFeatureVector.pkl"

# อ่านไฟล์ .pkl แล้วมาเก็บในตัวแปลงแบบ list
Train_dataset = pickle.load(open(train_path, 'rb'))
Test_dataset = pickle.load(open(test_path, 'rb'))

#Method เพื่อ set ข้อมูลให้กับ x_train, y_train, x_test, y_test
def setData(dataset):
    x = []
    y = []
    for index, value in enumerate(dataset):
        x.append( dataset[index][ : len(dataset[index])-1] )
        y.append( dataset[index][len(dataset[index])-1] )
    return x, y

# เก็บข้อมูล feature vector(x) และ brand(y) ของ feature นั้นๆ จากการเรียกใช้ Method setData
x_train, y_train = setData(Train_dataset)
x_test, y_test = setData(Test_dataset)

print('x_train:', len(x_train))
print('x_test:', len(x_test))
print('y_train:', len(y_train))
print('y_test:', len(y_test))

# train model หรือ สร้างต้นไม่ตัดสินใจ
model = DecisionTreeClassifier()
model = model.fit(x_train, y_train)
ypred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, ypred) * 100
matrix = confusion_matrix(y_test, ypred)
print("\nAccuracy:", accuracy)
print("Confusion matrix:\n", matrix)

# save model ไว้แบบไฟล์
file_name = 'ClassifierCarModel.pkl'
pickle.dump(model, open(file_name, 'wb'))
print("\nClassifierCarModel.pkl file saved.")