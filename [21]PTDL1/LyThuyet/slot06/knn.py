import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_excel("heart_short.xlsx")

# Xem dữ liệu
print(df)

# Input (đặc trưng) và Output (mục tiêu)
X = df[['t_i', 'c_i']]  # Đặc trưng
y = df['target']        # Mục tiêu


from sklearn.model_selection import train_test_split

# Chia tập dữ liệu (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train:\n", X_train)
print("X_test:\n", X_test)



from sklearn.neighbors import KNeighborsClassifier

# Khởi tạo mô hình KNN với k = 3
knnModel = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình trên tập train
knnModel.fit(X_train, y_train)
# Dự đoán
y_pred = knnModel.predict(X_test)

print("Giá trị thực tế X_test:", y_test.values)
print("Giá trị dự đoán X_test:", y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")
