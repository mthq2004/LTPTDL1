import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt 

# Đọc file 
df = pd.read_excel('data_bayes.xlsx', sheet_name="data")


# Tách dữ liệu thành 60:40 với random state = 16
from sklearn.model_selection import train_test_split
df_train, test_data = train_test_split(df, test_size=0.4, random_state=16)

print("Df Train:\n", df_train)
print("\nTest data:\n", test_data)


# Tập dữ liệu Train: Input (đặc trưng ) và Output (mục tiêu )
X_train = df_train[['ti', 'ci']].values 
y_train =df_train[['drug']].values  
X_train = X_train.astype(float)


# Huấn luyện 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)



