#!/usr/bin/env python3
"""
运行ZeoSyn数据集的基本可视化演示
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

print("正在加载ZeoSyn数据集...")
df = pd.read_excel('dataset/ZEOSYN.xlsx').drop(columns='Unnamed: 0')
df = df[~df['Si'].isna()].reset_index().drop(columns='index')
print(f"数据集加载完成，共 {len(df)} 条合成路线")

# 1. 可视化Si/Al比率分布
print("\n1. 可视化Si/Al比率分布...")
xlim = (0, 100)
plt.figure(figsize=(7, 7))
sns.histplot(df['Si']/df['Al'], bins=30, binrange=xlim, kde=True)
plt.xlabel('Si/Al ratio', fontsize=20)
plt.ylabel('Num. synthesis routes', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(*xlim)
plt.tight_layout()
plt.savefig('demo_si_al_ratio.png', dpi=150, bbox_inches='tight')
print("   保存为 demo_si_al_ratio.png")
plt.close()

# 2. 可视化结晶温度分布
print("\n2. 可视化结晶温度分布...")
xlim = (50, 220)
plt.figure(figsize=(7, 7))
sns.histplot(df['cryst_temp'], bins=30, binrange=xlim, kde=True)
plt.xlabel('Crystallization temperature (°C)', fontsize=20)
plt.ylabel('Num. synthesis routes', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(*xlim)
plt.tight_layout()
plt.savefig('demo_cryst_temp.png', dpi=150, bbox_inches='tight')
print("   保存为 demo_cryst_temp.png")
plt.close()

# 3. 显示最常见的OSDA
print("\n3. 显示最常见的OSDA...")
osda_counts = df['osda1 smiles'].value_counts()
print(f"   前10个最常见的OSDA:")
print(osda_counts.head(10))

# 可视化前10个OSDA分子
print("\n   可视化前10个OSDA分子...")
osdas = [Chem.MolFromSmiles(x) for x in osda_counts.keys()[:10]]
img = Draw.MolsToGridImage(osdas, molsPerRow=5, subImgSize=(200, 200))
img.save('demo_osda_molecules.png')
print("   保存为 demo_osda_molecules.png")

# 4. 显示最常见的沸石框架
print("\n4. 显示最常见的沸石框架...")
code1_counts = df['Code1'].value_counts()
print(f"   前10个最常见的沸石框架:")
print(code1_counts.head(10))

# 可视化前10个沸石框架
print("\n   可视化前10个沸石框架...")
top_frameworks = code1_counts.head(10)
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_frameworks)), top_frameworks.values)
plt.xticks(range(len(top_frameworks)), top_frameworks.index, rotation=45, ha='right')
plt.xlabel('Zeolite Framework', fontsize=16)
plt.ylabel('Number of Synthesis Routes', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('demo_top_frameworks.png', dpi=150, bbox_inches='tight')
print("   保存为 demo_top_frameworks.png")
plt.close()

# 5. 显示数据集统计信息
print("\n5. 数据集统计信息:")
print(f"   总合成路线数: {len(df)}")
print(f"   唯一沸石框架数: {df['Code1'].nunique()}")
print(f"   唯一OSDA数: {df['osda1 smiles'].nunique()}")
print(f"   结晶温度范围: {df['cryst_temp'].min():.1f} - {df['cryst_temp'].max():.1f} °C")
print(f"   结晶时间范围: {df['cryst_time'].min():.1f} - {df['cryst_time'].max():.1f} h")

print("\n✅ 演示完成！生成的图片:")
print("   - demo_si_al_ratio.png")
print("   - demo_cryst_temp.png")
print("   - demo_osda_molecules.png")
print("   - demo_top_frameworks.png")
