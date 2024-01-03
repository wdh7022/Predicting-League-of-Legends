from collections import Counter
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate 
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
RANDOM_SEED = 2020 
csv_data = r'D:\WXWork\1688855117494636\Cache\File\2023-11\high_diamond_ranked_10min.csv' 
data_df = pd.read_csv(csv_data, sep=',') 
data_df = data_df.drop(columns='gameId') 
data_df.describe() 
data_df.head(5)
drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin'] 
df = data_df.drop(columns=drop_features) 
info_names = [c[3:] for c in df.columns if c.startswith('red')] 
for info in info_names: 
    df['br' + info] = df['blue' + info] - df['red' + info] 
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood']) 
df.head(5)
df.describe()

discrete_df = df.copy() 
for c in df.columns[1:]: 
    # 使用pandas的cut函数将值划分为相等的分位
    discrete_df[c] = pd.cut(discrete_df[c], bins=5, labels=False)
all_y = discrete_df['blueWins'].values 
feature_names = discrete_df.columns[1:] 
all_x = discrete_df[feature_names].values 
print(all_x)

x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape 
print(x_train)

# LR = LogisticRegression(random_state=RANDOM_SEED, verbose=1, max_iter=1000) 
# LR.fit(x_train, y_train) 
# p_test = LR.predict(x_test) 
# test_acc = accuracy_score(p_test, y_test) 
# print('accuracy: {:.4}'.format(test_acc)) 
# DT = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=5, min_samples_split=10)
# DT.fit(x_train, y_train)  # 在训练集上训练

# p_test_dt = DT.predict(x_test)  # 在测试集上预测
# test_acc_dt = accuracy_score(p_test_dt, y_test)  # 计算准确率

# print('Decision Tree Accuracy: {:.4f}'.format(test_acc_dt))  # 输出决
SVM = SVC(random_state=RANDOM_SEED)
SVM.fit(x_train, y_train)  # 在训练集上训练

p_test_svm = SVM.predict(x_test)  # 在测试集上预测
test_acc_svm = accuracy_score(p_test_svm, y_test)  # 计算准确率

print('SVM Accuracy: {:.4f}'.format(test_acc_svm))  #