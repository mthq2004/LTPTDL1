import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_excel("lophoc.xlsx", sheet_name='data')


# Tạo dữ liệu Input (đặc trưng) và Output (Mục tiêu)
X = df.iloc[:, :-1]

y = df.iloc[:, -1]

# Gán nhãn 

from sklearn.preprocessing import LabelEncoder
enHealth = LabelEncoder()
X['Sức khỏe']= enHealth.fit_transform(X['Sức khỏe'])
enWeather = LabelEncoder()
X['Thời tiết']= enWeather.fit_transform(X['Thời tiết'])


# Trước khi đưa vào mô hình cắt dữ liệu làm đôi
X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
modelDT = DecisionTreeClassifier(criterion="entropy", random_state=34)
modelDT = modelDT.fit(X_train, y_train)

# Vẽ descision tree 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(
    modelDT,
    feature_names=['Sức khỏe', 'Thời tiết'],
    class_names=[ 'học','nghỉ' ],
    filled=True,
    fontsize=12
)
plt.title("Cây quyết định không giới hạn độ sâu", fontsize=16)
plt.show()

# Đánh giá mô hình 

# a. Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = modelDT.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# b. Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# c. Báo cáo phân loại
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['học', 'nghỉ']))


# d. Độ sâu của cây
print("Tree Depth:", modelDT.get_depth())
print("Number of Leaves:", modelDT.get_n_leaves())