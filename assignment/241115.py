import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# 1. 데이터 준비
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[
                       iris['data'], iris['target']], columns=iris['feature_names'] + ['species'])

# Versicolor (species=1), Virginica (species=2) 데이터만 필터링
iris_df = iris_df[iris_df['species'].isin([1, 2])]
iris_df = iris_df.reset_index(drop=True)  # 인덱스 초기화

# 컬럼 이름 간단히 변경
iris_df.columns = ['sl', 'sw', 'pl', 'pw', 'species']

# Y값 조정: Versicolor=0, Virginica=1
iris_df['species'] = iris_df['species'] - 1

# X와 Y 데이터 준비
X = iris_df[['sl', 'sw', 'pl', 'pw']]
y = iris_df['species']


# 2. 규제가 있는 경우 (기본: L2 Regularization)
clf_l2 = LogisticRegression()  # 기본: L2 정규화
clf_l2.fit(X, y)               # 학습
y_pred_l2 = clf_l2.predict(X)  # 예측
y_prob_l2 = clf_l2.predict_proba(X)[:, 1]  # y=1일 확률


# 결과 출력
print("=== L2 Regularization (Default) ===")
print("\nClassification Report (L2):")
print(classification_report(y, y_pred_l2))
print("\nConfusion Matrix (L2):")
print(confusion_matrix(y, y_pred_l2))


# 3. 규제가 없는 경우 (penalty=None)
clf_none = LogisticRegression(
    penalty=None, solver='lbfgs', max_iter=200)  # 규제 없음
clf_none.fit(X, y)              # 학습
y_pred_none = clf_none.predict(X)  # 예측
y_prob_none = clf_none.predict_proba(X)[:, 1]  # y=1일 확률


# 결과 출력
print("\n=== No Regularization (penalty=None) ===")
print("\nClassification Report (No Regularization):")
print(classification_report(y, y_pred_none))
print("\nConfusion Matrix (No Regularization):")
print(confusion_matrix(y, y_pred_none))
