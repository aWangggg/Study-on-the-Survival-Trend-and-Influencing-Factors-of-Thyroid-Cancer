import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import matplotlib

# 使用非GUI后端
matplotlib.use('Agg')

# 读取数据
data = pd.read_excel('ThyroidCancer.xlsx')

# 定义数值转换字典
conversion_dict = {
    'Age': {
        '00 years': 1, '01-04 years': 2, '05-09 years': 3,
        '10-14 years': 4, '15-19 years': 5, '20-24 years': 6, '25-29 years': 7,
        '30-34 years': 8, '35-39 years': 9, '40-44 years': 10, '45-49 years': 11,
        '50-54 years': 12, '55-59 years': 13, '60-64 years': 14, '65-69 years': 15,
        '70-74 years': 16, '75-79 years': 17, '80-84 years': 18, '85+ years': 19
    },
    'Sex': {'Male': 0, 'Female': 1},
    'Race recode (W, B, AI, API)': {'Asian or Pacific Islander': 0, 'Black': 1, 'White': 2},
    'Grade Pathological (2018+)': {'2': 2, '3': 3, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13},
    'RX Summ--Surg/Rad Seq': {
        'No radiation and/or no surgery; unknown if surgery and/or radiation given': 0,
        'Radiation after surgery': 1
    },
    'Radiation recode': {
        'Beam radiation': 0, 'Radioisotopes (1988+)': 1, 'None/Unknown': 2,
        'Radiation, NOS method or source not specified': 3,
        'Combination of beam with implants or isotopes': 4,
        'Radioactive implants (includes brachytherapy) (1988+)': 5,
        'Recommended, unknown if administered': 6, 'Refused (1988+)': 7
    },
    'Chemotherapy recode (yes, no/unk)': {'yes': 1, 'no/unk': 0},
    'SEER Combined Mets at DX-bone (2010+)': {'N/A': 0, 'Unknown': 1, 'Yes': 2, 'No': 3},
    'SEER Combined Mets at DX-brain (2010+)': {'N/A': 0, 'Unknown': 1, 'Yes': 2, 'No': 3},
    'SEER Combined Mets at DX-liver (2010+)': {'N/A': 0, 'Unknown': 1, 'Yes': 2, 'No': 3},
    'SEER Combined Mets at DX-lung (2010+)': {'N/A': 0, 'Unknown': 1, 'Yes': 2, 'No': 3},
    'Marital status at diagnosis': {
        'Married (including common law)': 0, 'Widowed': 1, 'Single (never married)': 2,
        'Divorced': 3, 'Separated': 4, 'Unknown': 5, 'Unmarried or Domestic Partner': 6
    },
    'Time from diagnosis to treatment in days recode': {'731+days': 731, 'Unable to calculate': np.nan},
    'Tumor Size Over Time Recode (1988+)': {
        'Unknown or size unreasonable (includes any tumor sizes 401-989)': np.nan,
        '990 (microscopic focus)': 990, '000 (no evidence of primary tumor)': 0
    }
}

# 将文本类型的特征转换为数值
for col, mapping in conversion_dict.items():
    if col in data.columns:
        data[col] = data[col].map(mapping).fillna(data[col])

# 填充缺失值
# 对数值型特征使用中位数填充
for col in data.select_dtypes(include=[np.number]).columns:
    data[col].fillna(data[col].median(), inplace=True)

# 对分类型特征使用众数填充
for col in data.select_dtypes(include=[object]).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# 定义特征和标签
feature_columns = [
    'Age', 'Sex', 'Year of diagnosis', 'Race recode (W, B, AI, API)',
    'Grade Pathological (2018+)', 'RX Summ--Surg/Rad Seq', 'Radiation recode',
    'Chemotherapy recode (yes, no/unk)', 'Time from diagnosis to treatment in days recode',
    'Tumor Size Over Time Recode (1988+)', 'Tumor Size Summary (2016+)',
    'Regional nodes examined (1988+)', 'Regional nodes positive (1988+)',
    'SEER Combined Mets at DX-bone (2010+)', 'SEER Combined Mets at DX-brain (2010+)',
    'SEER Combined Mets at DX-liver (2010+)', 'SEER Combined Mets at DX-lung (2010+)',
    'CS tumor size (2004-2015)', 'CS extension (2004-2015)', 'Marital status at diagnosis'
]

# 提取生存时间和生存状态
T = data['Survival Time']
E = (data['Year of death recode'] > 0).astype(int)  # 将死亡年份大于0的记录视为事件发生

# 删除缺失值过多的列，阈值设置为50%
threshold = len(data) * 0.5
filtered_feature_columns = [col for col in feature_columns if data[col].isnull().sum() <= threshold]

print(f"过滤后的特征列：{filtered_feature_columns}")

# 保存所有变量的Cox回归结果
summary_list = []

# 逐一对每个变量进行Cox回归分析
for col in filtered_feature_columns:
    df = pd.DataFrame({
        col: data[col],
        'T': T,
        'E': E
    })

    # 将分类变量转换为独热编码
    df = pd.get_dummies(df, drop_first=True)

    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col='T', event_col='E')

        # 保存每个变量的结果
        summary = cph.summary
        summary['variable'] = col
        summary_list.append(summary)

        # 绘制并保存图表
        plt.figure(figsize=(10, 6))
        cph.plot()
        plt.title(f'Cox Regression for {col}')
        plt.tight_layout()
        plt.savefig(f'CoxPH_Regression_{col.replace(" ", "_").replace("/", "_")}.png')
        plt.close()
    except Exception as e:
        print(f"Error in processing {col}: {e}")

# 将所有结果汇总为一个DataFrame并保存为CSV文件
all_summaries = pd.concat(summary_list)
all_summaries.to_csv('CoxPH_Regression_Summaries.csv', index=False)

print("Cox回归分析已完成，结果已保存。")
