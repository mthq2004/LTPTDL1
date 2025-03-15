import pandas as pd
df = pd.read_excel("lophoc.xlsx")
df
import math
p1 = 4/8
p2 = 4/8
entropy = -(p1 * math.log2(p1)) - (p2 * math.log2(p2))

print("Entropy:", entropy)
# Tính ENtroppy 
import math


entropyR = -0.918 * 3/7 - 1 * 2/7

print("Entropy:", entropyR)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_excel("lophoc.xlsx", sheet_name='data')
df
# Tạo dữ liệu Input (đặc trưng) và Output (Mục tiêu)
X = df.iloc[:, :-1]

y = df.iloc[:, -1]

X
from sklearn.preprocessing import LabelEncoder
enHealth = LabelEncoder()

X['Sức khỏe']= enHealth.fit_transform(X['Sức khỏe'])

enWeather = LabelEncoder()
X['Thời tiết']= enWeather.fit_transform(X['Thời tiết'])
X
# Trước khi đưa vào mô hình cắt dữ liệu làm đôi
X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

modelDT = DecisionTreeClassifier(criterion="entropy", max_depth=3)
modelDT = modelDT.fit(X_train, y_train)
modelDT
# Phân lớp: Học nghỉ
modelDT.classes_
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

feature_cols = X_train.columns
feature_cols

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Convert feature columns to a list of strings
feature_cols = list(X_train.columns.astype(str))

# Convert class names to a list of strings
class_names = list(modelDT.classes_.astype(str))

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(modelDT, feature_names=feature_cols, 
          class_names=class_names, fontsize=12, filled=True)

plt.show()
# Mua: 0
# Nang: 1
# U am: 2
# Chieu cao cay: 3

