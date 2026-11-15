#!/usr/bin/env python3
"""
验证模型在测试集上的准确率
确保预处理和数据划分与训练时完全一致
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from utils import cols_to_drop, osda_cols, X_cols, rename_disordered_interrupted

print("=" * 60)
print("验证模型在测试集上的准确率")
print("=" * 60)

# 1. 加载数据
print("\n1. 加载数据...")
df = pd.read_excel('dataset/ZEOSYN.xlsx')
df_osda = pd.read_csv('dataset/osda_descriptors.csv').drop(columns='Unnamed: 0')

# 2. 预处理（与训练时完全相同）
print("\n2. 预处理数据...")

# 获取唯一OSDA
unique_osdas = set(df['osda1 smiles'].value_counts().keys()) | \
               set(df['osda2 smiles'].value_counts().keys()) | \
               set(df['osda3 smiles'].value_counts().keys())

# 检查哪些OSDA有特征描述
osda_feat_avail = []
osda_feat_unavail = []
for osda in unique_osdas:
    if osda in list(df_osda['osda smiles']):
        osda_feat_avail.append(osda)
    else:
        osda_feat_unavail.append(osda)

print(f"   有特征的OSDA: {len(osda_feat_avail)}")
print(f"   无特征的OSDA: {len(osda_feat_unavail)}")

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
print(f"   类别数量: {y.nunique()}")

# 3. 划分训练集和测试集（与训练时完全相同）
print("\n3. 划分训练集和测试集...")
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
print(f"   训练集: {len(X_train)}")
print(f"   测试集: {len(X_test)}")

# 4. 加载模型
print("\n4. 加载模型...")
model = joblib.load('random_forest_model.pkl')
print("   ✅ 模型加载成功")

# 5. 预测测试集
print("\n5. 预测测试集...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 6. 计算准确率
print("\n6. 计算准确率...")
accuracy = accuracy_score(y_test, y_pred)
print(f"   整体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 7. 详细分类报告
print("\n7. 详细分类报告:")
report = classification_report(y_test, y_pred, output_dict=True)
print(f"   Macro avg precision: {report['macro avg']['precision']:.4f}")
print(f"   Macro avg recall: {report['macro avg']['recall']:.4f}")
print(f"   Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")

# 8. 显示前10个类别的性能
print("\n8. 前10个类别的性能:")
top_classes = y.value_counts().head(10).index
for cls in top_classes:
    if cls in report:
        print(f"   {cls}: precision={report[cls]['precision']:.4f}, "
              f"recall={report[cls]['recall']:.4f}, "
              f"f1={report[cls]['f1-score']:.4f}, "
              f"support={report[cls]['support']}")

# 9. 生成混淆矩阵（针对常见框架）
print("\n9. 生成混淆矩阵...")
common_frameworks = y.value_counts().head(10).index
mask_common = y_test.isin(common_frameworks)

y_test_common = y_test[mask_common]
y_pred_common = y_pred[mask_common]

cm = confusion_matrix(y_test_common, y_pred_common, labels=common_frameworks)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 保存混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=common_frameworks, yticklabels=common_frameworks)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Normalized Confusion Matrix (Top 10 Frameworks)', fontsize=16)
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   保存混淆矩阵为 test_confusion_matrix.png")
plt.close()

# 10. 特定框架的性能
print("\n10. 特定框架的性能:")
for target_zeo in ['LTA', 'MFI', 'CHA', '*BEA', 'FAU', 'Failed']:
    if target_zeo in report:
        print(f"   {target_zeo}: "
              f"precision={report[target_zeo]['precision']:.4f}, "
              f"recall={report[target_zeo]['recall']:.4f}, "
              f"f1={report[target_zeo]['f1-score']:.4f}, "
              f"support={report[target_zeo]['support']}")

print("\n" + "=" * 60)
print("✅ 验证完成!")
print("=" * 60)
print(f"\n总结:")
print(f"  - 测试集大小: {len(X_test)}")
print(f"  - 整体准确率: {accuracy*100:.2f}%")
print(f"  - Macro avg F1: {report['macro avg']['f1-score']:.4f}")
print(f"\n生成的文件:")
print(f"  - test_confusion_matrix.png")
