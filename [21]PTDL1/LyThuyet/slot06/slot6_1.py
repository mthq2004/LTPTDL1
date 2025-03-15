import math
a = math.sqrt((150-145)**(2)+(200-233)**(2))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel('heart_short.xlsx',sheet_name="data")
# tập dữ liệu Input (đăc trưng) và Output (mục tiêu)
X = df[['t_i','c_i']].values # input
y = df[['target']].values # output

X = X.astype(float)
y = y.astype(float)
print(f"X = {X}")
print(f'y = {y}')
# Sipiting data set: 7 mẫu train và 7 mẫu test (tỉ lệ 60:40)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,y,df.index,test_size=0.4, random_state=34)

# Index số mấy nằm trong tập test
indices_test
# ==> 0, 2, 7, 8, 11
from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_neighbors=5)

knnModel.fit(X_train, y_train)

# Dự báo mô hình trên tập X_test
y_test_predicted =knnModel.predict(X_test)
y_test_predicted
# ==> có 5 mẫu đánh giá chỉ đúng 1 mẫu nên mô hinhg chỉ đạt 20%
