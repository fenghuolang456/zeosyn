#!/usr/bin/env python3
"""
从ZeoSyn数据集中找到实际的LTA合成配方，并用模型预测验证
"""

import numpy as np
import pandas as pd
import joblib
from utils import X_cols, osda_cols, cols_to_drop, rename_disordered_interrupted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print("=" * 60)
print("从数据集中找到LTA合成配方并验证模型")
print("=" * 60)

# 1. 加载数据
print("\n1. 加载数据...")
df = pd.read_excel('dataset/ZEOSYN.xlsx')

# 2. 加载OSDA描述符
df_osda = pd.read_csv('dataset/osda_descriptors.csv').drop(columns='Unnamed: 0')

# 3. 筛选LTA合成路线
print("\n2. 筛选LTA合成路线...")
df_lta = df[df['Code1'] == 'LTA'].copy()
print(f"   数据集中LTA合成路线数: {len(df_lta)}")

# 4. 显示一些LTA合成配方
print("\n3. 显示前5个LTA合成配方:")
print("=" * 80)
for idx, row in df_lta.head(5).iterrows():
    print(f"\n配方 {idx}:")
    si_al = row['Si']/row['Al'] if row['Al'] != 0 else float('inf')
    print(f"  Si/Al: {row['Si']}/{row['Al']} = {si_al:.2f}")
    print(f"  Na: {row['Na']}, K: {row['K']}, Ca: {row['Ca']}")
    print(f"  OH: {row['OH']}, H2O: {row['H2O']}")
    print(f"  结晶时间: {row['cryst_time']} h, 温度: {row['cryst_temp']} °C")
    if pd.notna(row['osda1 smiles']):
        print(f"  OSDA1: {row['osda1 smiles'][:50]}...")
    else:
        print(f"  OSDA: 无")
    print(f"  DOI: {row['doi']}")
    print(f"  年份: {row['year']}")

# 5. 加载模型
print("\n4. 加载模型...")
model = joblib.load('random_forest_model.pkl')
print("   ✅ 模型加载成功")

# 6. 预处理数据（与训练时相同）
print("\n5. 预处理数据...")

# 获取唯一OSDA
unique_osdas = set(df['osda1 smiles'].value_counts().keys()) | \
               set(df['osda2 smiles'].value_counts().keys()) | \
               set(df['osda3 smiles'].value_counts().keys())

# 检查哪些OSDA有特征描述
osda_feat_avail = []
for osda in unique_osdas:
    if osda in list(df_osda['osda smiles']):
        osda_feat_avail.append(osda)

# 筛选有OSDA特征的合成路线
df_with_osda_feats = df[df['osda1 smiles'].isin(osda_feat_avail) |
                       df['osda2 smiles'].isin(osda_feat_avail) |
                       df['osda3 smiles'].isin(osda_feat_avail)]

# 添加无OSDA的合成路线
df_osda_free = df[(df['osda1 smiles'].isna()) &
                  (df['osda2 smiles'].isna()) &
                  (df['osda3 smiles'].isna()) &
                  (~df['Si'].isna())]

df_with_osda_feats = pd.concat([df_with_osda_feats, df_osda_free])

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

# Drop columns
X = df_syn_osda.drop(columns=cols_to_drop)

# Imputation for cryst_time and cryst_temp
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

print(f"   预处理后数据量: {len(X)}")
print(f"   特征数量: {len(X.columns)}")

# 7. 筛选LTA样本
print("\n6. 筛选LTA样本...")
lta_mask = y == 'LTA'
X_lta = X[lta_mask]
y_lta = y[lta_mask]
print(f"   LTA样本数: {len(X_lta)}")

# 8. 用模型预测LTA样本
print("\n7. 用模型预测LTA样本...")
y_pred_lta = model.predict(X_lta)
y_prob_lta = model.predict_proba(X_lta)

# 9. 计算预测准确率
correct_predictions = (y_pred_lta == 'LTA').sum()
accuracy = correct_predictions / len(X_lta)
print(f"   正确预测为LTA的数量: {correct_predictions}/{len(X_lta)}")
print(f"   预测准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 10. 显示预测结果分布
print("\n8. 预测结果分布:")
pred_counts = pd.Series(y_pred_lta).value_counts()
print(pred_counts.head(10))

# 11. 显示LTA预测概率分布
lta_idx = np.where(model.classes_ == 'LTA')[0]
if len(lta_idx) > 0:
    lta_idx = lta_idx[0]
    lta_probs = y_prob_lta[:, lta_idx]
    
    print(f"\n9. LTA预测概率统计:")
    print(f"   平均概率: {lta_probs.mean():.4f}")
    print(f"   中位数: {np.median(lta_probs):.4f}")
    print(f"   最大值: {lta_probs.max():.4f}")
    print(f"   最小值: {lta_probs.min():.4f}")
    
    # 显示概率最高的5个样本
    top_indices = np.argsort(lta_probs)[-5:][::-1]
    print(f"\n10. LTA预测概率最高的5个样本:")
    for i in top_indices:
        print(f"    样本 {i}: LTA概率 = {lta_probs[i]:.4f} ({lta_probs[i]*100:.2f}%)")

# 12. 对比：随机选择其他框架的样本
print("\n11. 对比测试：随机选择其他框架的样本...")
other_mask = y != 'LTA'
other_indices = np.where(other_mask)[0]
random_indices = np.random.choice(other_indices, size=min(100, len(other_indices)), replace=False)

X_other = X.iloc[random_indices]
y_other = y.iloc[random_indices]
y_pred_other = model.predict(X_other)
y_prob_other = model.predict_proba(X_other)

lta_idx = np.where(model.classes_ == 'LTA')[0]
if len(lta_idx) > 0:
    lta_idx = lta_idx[0]
    lta_probs_other = y_prob_other[:, lta_idx]
    print(f"   其他框架样本的LTA预测概率:")
    print(f"   平均概率: {lta_probs_other.mean():.4f}")
    print(f"   中位数: {np.median(lta_probs_other):.4f}")
    print(f"   最大值: {lta_probs_other.max():.4f}")

print("\n" + "=" * 60)
print("✅ 验证完成!")
print("=" * 60)
print(f"\n总结:")
print(f"  - LTA合成路线数: {len(X_lta)}")
print(f"  - 预测为LTA的数量: {correct_predictions}")
print(f"  - 预测准确率: {accuracy*100:.2f}%")
print(f"  - LTA平均预测概率: {lta_probs.mean():.4f}")
