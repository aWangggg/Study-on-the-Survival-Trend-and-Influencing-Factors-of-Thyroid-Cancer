import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re

# 使用非GUI后端
matplotlib.use('Agg')

# 读取数据
data = pd.read_excel('ThyroidCancer.xlsx')

# 定义年份列和淋巴转移列
year_column = 'Year of diagnosis'
lymph_node_metastasis_columns = [
    'Regional nodes positive (1988+)',
    'SEER Combined Mets at DX-bone (2010+)',
    'SEER Combined Mets at DX-brain (2010+)',
    'SEER Combined Mets at DX-liver (2010+)',
    'SEER Combined Mets at DX-lung (2010+)',
]

# 删除年份列中的缺失值
data = data.dropna(subset=[year_column])

# 确保年份列是整数类型
data[year_column] = data[year_column].astype(int)

# 将淋巴转移列转换为数值类型，非数值转为NaN，并填充为0
for column in lymph_node_metastasis_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0)

# 计算每年的总病例数
total_cases_per_year = data[year_column].value_counts().sort_index()

# 计算每年的淋巴转移病例数
lymph_node_metastasis_cases_per_year = data[data[lymph_node_metastasis_columns].sum(axis=1) > 0][year_column].value_counts().sort_index()

# 计算淋巴转移率
lymph_node_metastasis_rate_per_year = (lymph_node_metastasis_cases_per_year / total_cases_per_year) * 100

# 绘制总发病率趋势图
plt.figure(figsize=(12, 6))
total_cases_per_year.plot(kind='bar', color='skyblue')
plt.title('Total Incidence of Thyroid Cancer Over Time')
plt.xlabel('Year of Diagnosis')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Total_Incidence_Trend.png')
plt.close()

# 绘制淋巴转移率趋势图
plt.figure(figsize=(12, 6))
lymph_node_metastasis_rate_per_year.plot(kind='bar', color='salmon')
plt.title('Lymph Node Metastasis Rate of Thyroid Cancer Over Time')
plt.xlabel('Year of Diagnosis')
plt.ylabel('Metastasis Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Lymph_Node_Metastasis_Rate_Trend.png')
plt.close()

print("时间趋势图已生成并保存。")
