import pandas as pd
import numpy as np
# 读取数据
data = pd.read_excel('ThyroidCancer.xlsx')

# 定义数值转换字典
conversion_dict = {
    'Age': {
        '00 years': 0, '01-04 years': 2, '05-09 years': 7,
        '10-14 years': 12, '15-19 years': 17, '20-24 years': 22, '25-29 years': 27,
        '30-34 years': 32, '35-39 years': 37, '40-44 years': 42, '45-49 years': 47,
        '50-54 years': 52, '55-59 years': 57, '60-64 years': 62, '65-69 years': 67,
        '70-74 years': 72, '75-79 years': 77, '80-84 years': 82, '85+ years': 87
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
    'Time from diagnosis to treatment in days recode': {'731+days': 731, 'Unable to calculate': None},
    'Tumor Size Over Time Recode (1988+)': {
        'Unknown or size unreasonable (includes any tumor sizes 401-989)': None,
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

# 描述性统计分析
desc_stats = data.describe(include='all')

# 保存描述性统计结果
desc_stats.to_csv('Descriptive_Statistics.csv')

print("描述性统计分析已完成，结果已保存为 Descriptive_Statistics.csv。")
