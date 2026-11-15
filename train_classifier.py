#!/usr/bin/env python3
"""
训练ZeoSyn沸石框架预测分类器
基于论文中的随机森林模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import rename_disordered_interrupted, cols_to_drop, osda_cols, X_cols

print("=" * 60)
print("ZeoSyn 沸石框架预测分类器训练")
print("=" * 60)

# 1. 加载数据
print("\n1. 加载数据集...")
df = pd.read_excel('dataset/ZEOSYN.xlsx')
print(f"   总数据量: {len(df)} 条合成路线")

# 2. 加载OSDA描述符
print("\n2. 加载OSDA描述符...")
df_osda = pd.read_csv('dataset/osda_descriptors.csv').drop(columns='Unnamed: 0')
print(f"   OSDA描述符数量: {len(df_osda)}")

# 3. 获取唯一OSDA
print("\n3. 处理OSDA特征...")
unique_osdas = set(df['osda1 smiles'].value_counts().keys()) | \
               set(df['osda2 smiles'].value_counts().keys()) | \
               set(df['osda3 smiles'].value_counts().keys())
print(f"   唯一OSDA总数: {len(unique_osdas)}")

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

# 4. 筛选有OSDA特征的合成路线
print("\n4. 筛选数据...")
df_with_osda_feats = df[df['osda1 smiles'].isin(osda_feat_avail) |
                       df['osda2 smiles'].isin(osda_feat_avail) |
                       df['osda3 smiles'].isin(osda_feat_avail)]
print(f"   有OSDA特征的合成路线: {len(df_with_osda_feats)}")

# 添加无OSDA的合成路线
df_osda_free = df[(df['osda1 smiles'].isna()) &
                  (df['osda2 smiles'].isna()) &
                  (df['osda3 smiles'].isna()) &
                  (~df['Si'].isna())]
print(f"   无OSDA的合成路线: {len(df_osda_free)}")

df_with_osda_feats = pd.concat([df_with_osda_feats, df_osda_free])
print(f"   总合成路线: {len(df_with_osda_feats)}")

# 5. 选择OSDA特征列
print("\n5. 选择OSDA特征列...")
dfs_with_osda_cols = []
for osda_col in osda_cols:
    df_with_osda_col = df_osda.loc[:, df_osda.columns.str.contains(osda_col)]
    dfs_with_osda_cols.append(df_with_osda_col)
dfs_with_osda_cols.append(df_osda.loc[:, df_osda.columns.str.contains('osda smiles')])

df_selected_osda_cols = pd.concat(dfs_with_osda_cols, axis=1)
print(f"   选择的OSDA特征列数: {len(df_selected_osda_cols.columns)}")

# 6. 合并OSDA特征
print("\n6. 合并OSDA特征...")
# 添加OSDA1
df_syn_osda = df_with_osda_feats.merge(
    df_selected_osda_cols.add_prefix('osda1_'),
    left_on='osda1 smiles',
    right_on='osda1_osda smiles',
    how='outer'
)
df_syn_osda = df_syn_osda[~df_syn_osda['Unnamed: 0'].isna()]

# 添加OSDA2
df_syn_osda = df_syn_osda.merge(
    df_selected_osda_cols.add_prefix('osda2_'),
    left_on='osda2 smiles',
    right_on='osda2_osda smiles',
    how='outer'
)
df_syn_osda = df_syn_osda[~df_syn_osda['Unnamed: 0'].isna()]

# 添加OSDA3
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
print(f"   合并后数据量: {len(df_syn_osda)}")

# 7. 数据预处理
print("\n7. 数据预处理...")
# 删除不需要的列
X = df_syn_osda.drop(columns=cols_to_drop)

# 对缺失值进行插补
print("   对缺失值进行插补...")
columns = list(X.columns)
imp = IterativeImputer(min_value=0., sample_posterior=True,
                       skip_complete=True, random_state=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X))
X.columns = columns

X_y = X
X_y['Code1'] = df_syn_osda['Code1']

# 空的Code1表示非晶/致密相 - 替换为'Failed'
X_y['Code1'] = X_y['Code1'].fillna('Failed')

# 定义X和y
X = X_y.drop(columns=['Code1'])
y = X_y['Code1']

# 特征选择
X = X[X_cols.keys()]
X.columns = X_cols.values()

print(f"   特征数量: {len(X.columns)}")
print(f"   有效数据量: {len(X)}")
print(f"   类别数量: {y.nunique()}")

# 8. 划分训练集和测试集
print("\n8. 划分训练集和测试集 (80/20)...")
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
print(f"   训练集: {len(X_train)}")
print(f"   测试集: {len(X_test)}")

# 9. 训练随机森林分类器
print("\n9. 训练随机森林分类器...")
print("   使用默认参数...")
rf = RandomForestClassifier(
    max_depth=None,
    random_state=random_state,
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1
)

rf.fit(X_train, y_train)
print("   ✅ 训练完成!")

# 10. 评估模型
print("\n10. 评估模型性能...")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 详细分类报告
print("\n   分类报告 (前20个类别):")
unique_classes = y_test.unique()
print(f"   总类别数: {len(unique_classes)}")
target_names = [str(c) for c in unique_classes[:20]]
report = classification_report(y_test, y_pred, target_names=target_names,
                                zero_division=0)
print(report)

# 11. 特征重要性
print("\n11. 计算特征重要性...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("   前15个最重要的特征:")
print(feature_importance.head(15).to_string(index=False))

# 可视化特征重要性
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Top 15 Feature Importances', fontsize=16)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("   保存特征重要性图为 feature_importance.png")
plt.close()

# 12. 混淆矩阵（针对常见框架）
print("\n12. 生成混淆矩阵...")
common_frameworks = y.value_counts().head(10).index
mask_common = y_test.isin(common_frameworks)

y_test_common = y_test[mask_common]
y_pred_common = y_pred[mask_common]

cm = confusion_matrix(y_test_common, y_pred_common, labels=common_frameworks)

# 归一化混淆矩阵
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=common_frameworks, yticklabels=common_frameworks)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Normalized Confusion Matrix (Top 10 Frameworks)', fontsize=16)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   保存混淆矩阵图为 confusion_matrix.png")
plt.close()

# 13. 保存模型
print("\n13. 保存模型...")
import joblib
joblib.dump(rf, 'random_forest_model.pkl')
print("   模型保存为 random_forest_model.pkl")

print("\n" + "=" * 60)
print("✅ 训练完成!")
print("=" * 60)
print(f"\n总结:")
print(f"  - 准确率: {accuracy*100:.2f}%")
print(f"  - 类别数: {len(unique_classes)}")
print(f"  - 训练集大小: {len(X_train)}")
print(f"  - 测试集大小: {len(X_test)}")
print(f"\n生成的文件:")
print(f"  - feature_importance.png")
print(f"  - confusion_matrix.png")
print(f"  - random_forest_model.pkl")
