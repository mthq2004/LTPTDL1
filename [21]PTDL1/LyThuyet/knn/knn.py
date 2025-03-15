import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('heart_short.xlsx',sheet_name="data")

df
#  Tách dữ liệu thành input và output
X = df[['t_i','c_i']].values # input
y = df[['target']].values # output

X = X.astype(float)
y = y.astype(float)
print("Data X: ", X , '\n')
print("Data Y: ", y)
# Chia tập dữ liệu thành training và testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,y,df.index,test_size=0.4, random_state=34)
X_train




# Sipiting data set: 7 mẫu train và 7 mẫu test (tỉ lệ 60:40)
# Câu hỏi: cho biết tập test có bao nhiêu mẫu dữ liệu 
# hãy cho biết index của những dòng dữ liệu nào được lấy ngẫu nhiên vào tập test
# Yêu cầu: sinh viên cắt dữ liệu Input và OUtput của tập test vào Execl để quan sát
# Đề cho sẳn test_size , random_state
# Huấn luyện mô hình với k=5
from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(X_train, y_train)




# Cau hoi mo hinh k=5 voi mo hinh k=4 mo hinh nao ok hon?
# Sử dụng mô hình K-Nearest Neighbors với k=5 để dự đoán nhãn đầu ra.
# knnModel.fit(X_train, y_train): Huấn luyện mô hình dựa trên tập huấn luyện.
# B5: Đánh giá  - Du bao mo hinh qua tap X_test 
y_test_predicted = knnModel.predict(X_test)
print(f'Dự đoán 5 thằng được đi ra: {y_test_predicted}' )         
y_test_predicted = knnModel.predict(X_test)
print(f'Độ chính xác của mô hình là: {knnModel.score(X_test, y_test) *100} %')
# Câu hỏi: Tập giá trị nhãn để dự báo là gì? --> kết luận đây là mô hình nhị phân 
classes = knnModel.classes_
classes
# Tính xác suất tiền trước khi quyết định trên tập mẫu 
y_test_prob = knnModel.predict_proba(X_test)
y_test_prob



#           0   |  1 :  0 là  không bệnh , 1 là bệnh 
#     AN   :0.8 | 0.2 --> có 4 không bệnh - 1 bênh
#    Thanh :0.6 | 0.4 --> có 3 không bệnh - 2 bệnh


# Câu hỏi: Nếu thiết lập ngưỡng (threshold) quyết định là 0.85. Thì kết quả accuracy
print('prediction with threshold 0.85')
y_pred_test_085 = (knnModel.predict_proba(X_test)[:, 1] >= 0.85).astype(float)
y_pred_test_085
# Câu hỏi: Nếu thiết lập ngưỡng (threshold) quyết định là 0.25. Thì kết quả accuracy
print('prediction with threshold 0.25')
y_pred_test_085 = (knnModel.predict_proba(X_test)[:, 1] >= 0.25).astype(float)
y_pred_test_085
# array([0., 1., 1., 1., 1.]) bon so 1 la bon benh , 1 khong benh 
# Tạo confusion - Metric 
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_test_predicted)
cf_matrix

# Các giá trị tn, fp, fn , tp
tn, fp, fn , tp = confusion_matrix(y_test, y_test_predicted).ravel()
print(f'{tn}, {fp}, {fn}, {tp}')

#### Vẽ biểu đồ AUC-ROC:
from sklearn import metrics
y_pred_proba = knnModel.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,'go-',label="AUC="+str(auc))
plt.plot([0,1],[0,1],'r--')
plt.title("AUC & ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.fill_between(fpr, tpr, facecolor='lightgreen', alpha=0.7)
plt.show()
# Màu xanh dưới màu đỏ thì ---> Mô hình xấu 
# Màu xanh cao hơn lớn hơn màu đỏ --> Mô hình tốt 
# Tính precison, recall, F1 theo nhom
from sklearn.metrics import classification_report
target_names = ['Không bệnh', 'Có bệnh']
print(classification_report(y_test, y_test_predicted, target_names=target_names))
# Xây dụng chương tình dự báo

import pickle
pickle.dump(knnModel, open('model_KNN_Heart.sav', 'wb'))


import pickle
#Load model từ storage
loaded_model = pickle.load(open('model_KNN_Heart.sav', 'rb'))
v1 = float(input('t_i: '))
v2 = float(input('c_i: '))

y_pred = loaded_model.predict([[v1,v2]])
print('Kết quả dự báo bệnh tim: '+ str(y_pred[0]))

if (y_pred[0] == 1):
    print("Bị bệnh tim")
else:
    print("Không bị bệnh")
# t_i : 1.55
# c_i: 2.75






