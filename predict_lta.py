#!/usr/bin/env python3
"""
使用训练好的模型预测LTA沸石框架
"""

import numpy as np
import pandas as pd
import joblib
from utils import X_cols, osda_cols, cols_to_drop
from rdkit import Chem

print("=" * 60)
print("使用模型预测LTA沸石框架")
print("=" * 60)

# 1. 加载训练好的模型
print("\n1. 加载模型...")
try:
    model = joblib.load('random_forest_model.pkl')
    print("   ✅ 模型加载成功")
except FileNotFoundError:
    print("   ❌ 未找到模型文件，请先运行训练脚本")
    exit(1)

# 2. 定义一个示例合成配方（用于预测LTA）
print("\n2. 定义合成配方...")

# 示例：合成LTA沸石的典型配方
# 参考文献中的LTA合成条件
synthesis_recipe = {
    # 凝胶组成
    'Si': 1.0,        # 二氧化硅
    'Al': 0.25,       # 氧化铝
    'Na': 1.0,        # 氢氧化钠
    'K': 0.0,         # 氢氧化钾
    'Ca': 0.0,        # 氯化钙
    'OH': 1.0,        # 氢氧化根
    'H2O': 20.0,      # 水
    'sda1': 0.0,      # OSDA用量（无OSDA合成）

    # 反应条件
    'cryst_time': 24,  # 结晶时间（小时）
    'cryst_temp': 100, # 结晶温度（°C）

    # OSDA特征（无OSDA时全为0）
    'osda1_asphericity_mean_0': 0,
    'osda1_axes_mean_0': 0,
    'osda1_axes_mean_1': 0,
    'osda1_formal_charge': 0,
    'osda1_free_sasa_mean_0': 0,
    'osda1_mol_weight': 0,
    'osda1_npr1_mean_0': 0,
    'osda1_npr2_mean_0': 0,
    'osda1_num_rot_bonds_mean_0': 0,
    'osda1_pmi1_mean_0': 0,
    'osda1_pmi2_mean_0': 0,
    'osda1_pmi3_mean_0': 0,
    'osda1_spherocity_index_mean_0': 0,
    'osda1_volume_mean_0': 0,
}

# 创建DataFrame - 需要包含所有特征列
all_features = {
    # 凝胶组成
    'Si': 1.0, 'Al': 0.25, 'P': 0.0, 'Na': 1.0, 'K': 0.0, 'Li': 0.0,
    'Sr': 0.0, 'Rb': 0.0, 'Cs': 0.0, 'Ba': 0.0, 'Ca': 0.0,
    'F': 0.0, 'Ge': 0.0, 'Ti': 0.0, 'B': 0.0, 'Mg': 0.0, 'Ga': 0.0,
    'Zn': 0.0, 'Be': 0.0, 'W': 0.0, 'Cu': 0.0, 'Sn': 0.0, 'Zr': 0.0,
    'V': 0.0, 'H2O': 20.0, 'sda1': 0.0, 'OH': 1.0,

    # 反应条件
    'cryst_time': 24, 'cryst_temp': 100,

    # OSDA特征（无OSDA时全为0）
    'osda1_asphericity_mean_0': 0, 'osda1_axes_mean_0': 0,
    'osda1_axes_mean_1': 0, 'osda1_formal_charge': 0,
    'osda1_free_sasa_mean_0': 0, 'osda1_mol_weight': 0,
    'osda1_npr1_mean_0': 0, 'osda1_npr2_mean_0': 0,
    'osda1_num_rot_bonds_mean_0': 0, 'osda1_pmi1_mean_0': 0,
    'osda1_pmi2_mean_0': 0, 'osda1_pmi3_mean_0': 0,
    'osda1_spherocity_index_mean_0': 0, 'osda1_volume_mean_0': 0,
}

df = pd.DataFrame([all_features])

# 确保列顺序正确，并转换为模型期望的特征名
feature_cols = list(X_cols.keys())
X = df[feature_cols]

# 转换列名为模型期望的格式
X.columns = X_cols.values()

print("   合成配方:")
for col in X.columns:
    if col in ['Si', 'Al', 'Na', 'K', 'OH', 'H$_2$O', 'Crystallization time', 'Crystallization temp']:
        print(f"     {col}: {X[col].values[0]:.2f}")

# 3. 预处理（与训练时相同）
print("\n3. 预处理...")
# 使用训练时相同的插补器
# 注意：这里我们简化处理，实际应该加载训练时的插补器
# 对于这个示例，我们假设没有缺失值

# 4. 预测
print("\n4. 预测...")
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# 5. 获取预测结果
print("\n5. 预测结果:")
print(f"   预测框架: {predictions[0]}")

# 获取所有类别的概率
classes = model.classes_
print(f"\n   所有类别数量: {len(classes)}")

# 找到LTA的索引
lta_idx = np.where(classes == 'LTA')[0]
if len(lta_idx) > 0:
    lta_idx = lta_idx[0]
    lta_prob = probabilities[0][lta_idx]
    print(f"   LTA预测概率: {lta_prob:.4f} ({lta_prob*100:.2f}%)")
else:
    print("   ⚠️  LTA不在预测类别中")

# 6. 获取Top 10预测结果
print("\n6. Top 10预测结果:")
top_indices = np.argsort(probabilities[0])[-10:][::-1]
for i in top_indices:
    class_name = classes[i]
    prob = probabilities[0][i]
    print(f"   {class_name}: {prob:.4f} ({prob*100:.2f}%)")

# 7. 如果LTA不在Top 10，显示LTA的排名
top_10_classes = [classes[i] for i in top_indices]
if 'LTA' not in top_10_classes and len(lta_idx) > 0:
    lta_prob = probabilities[0][lta_idx[0]]
    # 计算LTA的排名
    rank = np.sum(probabilities[0] > lta_prob) + 1
    print(f"\n   LTA排名: 第{rank}位 (概率: {lta_prob:.4f})")

print("\n" + "=" * 60)
print("✅ 预测完成!")
print("=" * 60)

# 8. 尝试不同的合成条件来优化LTA合成
print("\n8. 尝试不同的合成条件来优化LTA合成...")

# 定义几个不同的合成条件
base_features = {
    'Si': 1.0, 'Al': 0.25, 'P': 0.0, 'Na': 1.0, 'K': 0.0, 'Li': 0.0,
    'Sr': 0.0, 'Rb': 0.0, 'Cs': 0.0, 'Ba': 0.0, 'Ca': 0.0,
    'F': 0.0, 'Ge': 0.0, 'Ti': 0.0, 'B': 0.0, 'Mg': 0.0, 'Ga': 0.0,
    'Zn': 0.0, 'Be': 0.0, 'W': 0.0, 'Cu': 0.0, 'Sn': 0.0, 'Zr': 0.0,
    'V': 0.0, 'H2O': 20.0, 'sda1': 0.0, 'OH': 1.0,
    'osda1_asphericity_mean_0': 0, 'osda1_axes_mean_0': 0,
    'osda1_axes_mean_1': 0, 'osda1_formal_charge': 0,
    'osda1_free_sasa_mean_0': 0, 'osda1_mol_weight': 0,
    'osda1_npr1_mean_0': 0, 'osda1_npr2_mean_0': 0,
    'osda1_num_rot_bonds_mean_0': 0, 'osda1_pmi1_mean_0': 0,
    'osda1_pmi2_mean_0': 0, 'osda1_pmi3_mean_0': 0,
    'osda1_spherocity_index_mean_0': 0, 'osda1_volume_mean_0': 0,
}

recipes = [
    {
        'name': '低温合成',
        'cryst_time': 48, 'cryst_temp': 80,
    },
    {
        'name': '标准合成',
        'cryst_time': 24, 'cryst_temp': 100,
    },
    {
        'name': '高温合成',
        'cryst_time': 12, 'cryst_temp': 120,
    },
]

for recipe in recipes:
    name = recipe.pop('name')
    # 合并基础特征和特定条件
    full_recipe = base_features.copy()
    full_recipe.update(recipe)
    df_test = pd.DataFrame([full_recipe])
    X_test = df_test[feature_cols]
    X_test.columns = X_cols.values()
    
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    
    # 获取LTA概率
    lta_idx = np.where(model.classes_ == 'LTA')[0]
    if len(lta_idx) > 0:
        lta_prob = prob[0][lta_idx[0]]
        print(f"   {name}: 预测={pred[0]}, LTA概率={lta_prob:.4f} ({lta_prob*100:.2f}%)")
    else:
        print(f"   {name}: 预测={pred[0]}, LTA不在类别中")
