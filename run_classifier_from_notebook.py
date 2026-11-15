#!/usr/bin/env python3
"""
直接运行classifier.ipynb中的训练代码
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import rename_disordered_interrupted, cols_to_drop, osda_cols, X_cols

print("=" * 60)
print("运行classifier.ipynb中的训练代码")
print("=" * 60)

# Load synthesis dataset
print("\n1. 加载数据...")
df = pd.read_excel('dataset/ZEOSYN.xlsx')

# Load osda descriptors
df_osda = pd.read_csv('dataset/osda_descriptors.csv').drop(columns='Unnamed: 0')

# 获取唯一OSDA
unique_osdas = set(df['osda1 smiles'].value_counts().keys()) | \
               set(df['osda2 smiles'].value_counts().keys()) | \
               set(df['osda3 smiles'].value_counts().keys())

print(f"唯一OSDA总数: {len(unique_osdas)}")

# 检查哪些OSDA有特征描述
osda_feat_avail = []
osda_feat_unavail = []
for osda in unique_osdas:
    if osda in list(df_osda['osda smiles']):
        osda_feat_avail.append(osda)
    else:
        osda_feat_unavail.append(osda)

print(f"有特征的OSDA: {len(osda_feat_avail)}")
print(f"无特征的OSDA: {len(osda_feat_unavail)}")

# 筛选有OSDA特征的合成路线
df_with_osda_feats = df[df['osda1 smiles'].isin(osda_feat_avail) |
                       df['osda2 smiles'].isin(osda_feat_avail) |
                       df['osda3 smiles'].isin(osda_feat_avail)]
print(f"有OSDA特征的合成路线: {len(df_with_osda_feats)}")

# 添加无OSDA的合成路线
df_osda_free = df[(df['osda1 smiles'].isna()) &
                  (df['osda2 smiles'].isna()) &
                  (df['osda3 smiles'].isna()) &
                  (~df['Si'].isna())]
print(f"无OSDA的合成路线: {len(df_osda_free)}")

df_with_osda_feats = pd.concat([df_with_osda_feats, df_osda_free])
print(f"总合成路线: {len(df_with_osda_feats)}")

# 选择OSDA特征列
dfs_with_osda_cols = []
for osda_col in osda_cols:
    df_with_osda_col = df_osda.loc[:, df_osda.columns.str.contains(osda_col)]
    dfs_with_osda_cols.append(df_with_osda_col)
dfs_with_osda_cols.append(df_osda.loc[:, df_osda.columns.str.contains('osda smiles')])

df_selected_osda_cols = pd.concat(dfs_with_osda_cols, axis=1)

# 合并OSDA特征
df_syn_osda = df_with_osda_feats.merge(
    df_selected_osda_cols.add_prefix('osda1_'),
    left_on='osda1 smiles',
    right_on='osda1_osda smiles',
    how='outer'
)
df_syn_osda = df_syn_osda[~df_syn_osda['Unnamed: 0'].isna()]

df_syn_osda = df_syn_osda.merge(
    df_selected_osda_cols.add_prefix('osda2_'),
    left_on='osda2 smiles',
    right_on='osda2_osda smiles',
    how='outer'
)
df_syn_osda = df_syn_osda[~df_syn_osda['Unnamed: 0'].isna()]

df_syn_osda = df_syn_osda.merge(
    df_selected_osda_cols.add_prefix('osda3_'),
    left_on='osda3 smiles',
    right_on='osda3_osda smiles',
    how='outer'
)
df_syn_osda = df_syn_osda[~df_syn_osda['Unnamed: 0'].isna()]

# 将OSDA特征中的NaN替换为0
for osda_col in osda_cols:
    df_syn_osda.loc[:, df_syn_osda.columns.str.contains(osda_col)] = \
        df_syn_osda.loc[:, df_syn_osda.columns.str.contains(osda_col)].fillna(0)

# 删除不需要的列
df_syn_osda = df_syn_osda.drop(columns=[
    'Unnamed: 0',
    'osda1_osda smiles',
    'osda2_osda smiles',
    'osda3_osda smiles'
])

print(f"合并后数据量: {len(df_syn_osda)}")

# Drop columns
X = df_syn_osda.drop(columns=cols_to_drop)

# Imputation for cryst_time and cryst_temp
print("\n2. 对缺失值进行插补...")
columns = list(X.columns)
imp = IterativeImputer(min_value=0., sample_posterior=True,
                       skip_complete=True, random_state=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X))
X.columns = columns

X_y = X
X_y['Code1'] = df_syn_osda['Code1']

# Empty 'Code1' element means amorphous/dense phases - replace with 'Failed'
X_y['Code1'] = X_y['Code1'].fillna('Failed')

# Define X and y
X = X_y.drop(columns=['Code1'])
y = X_y['Code1']

# Feature selection
X = X[X_cols.keys()]
X.columns = X_cols.values()

print(f"特征数量: {len(X.columns)}")
print(f"有效数据量: {len(X)}")
print(f"类别数量: {y.nunique()}")

# Training classifier
print("\n3. 训练分类器...")
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

model = RandomForestClassifier(max_depth=None, random_state=random_state, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# Calculate metrics with results
y_pred = model.predict(X_test)
results = classification_report(y_test, y_pred, output_dict=True)

print('\n======MODEL PERFORMANCE METRICS======')
print('acc: ', results['accuracy'])
print('macro avg:', results['macro avg'])

# 保存模型
print("\n4. 保存模型...")
joblib.dump(model, 'random_forest_model.pkl')
print("模型保存为 random_forest_model.pkl")

print("\n✅ 训练完成!")
