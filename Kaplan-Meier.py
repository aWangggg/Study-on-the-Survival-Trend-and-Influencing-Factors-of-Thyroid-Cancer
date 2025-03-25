import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib
import re

# 使用非GUI后端
matplotlib.use('Agg')

# 读取数据
data = pd.read_excel('ThyroidCancer.xlsx')

# 定义要分析的变量
variables = [
    'Age', 'Sex', 'Year of diagnosis', 'Race recode (W, B, AI, API)', 'Grade Pathological (2018+)',
    'RX Summ--Surg/Rad Seq', 'Radiation recode', 'Chemotherapy recode (yes, no/unk)',
    'Time from diagnosis to treatment in days recode', 'Tumor Size Over Time Recode (1988+)',
    'Tumor Size Summary (2016+)', 'Regional nodes examined (1988+)', 'Regional nodes positive (1988+)',
    'SEER Combined Mets at DX-bone (2010+)', 'SEER Combined Mets at DX-brain (2010+)',
    'SEER Combined Mets at DX-liver (2010+)', 'SEER Combined Mets at DX-lung (2010+)',
    'CS tumor size (2004-2015)', 'CS extension (2004-2015)', 'EOD 10 - size (1988-2003)',
    'Year of follow-up recode', 'Year of death recode', 'Survival Time', 'Marital status at diagnosis'
]

# 进行生存分析并保存图像
kmf = KaplanMeierFitter()

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

for var in variables:
    # 检查变量是否存在空值并删除这些行
    df = data.dropna(subset=[var, 'Survival Time'])

    # 分组变量
    groups = df[var].unique()

    plt.figure(figsize=(12, 8))

    for group in groups:
        # 选择某一组的数据
        ix = df[var] == group
        kmf.fit(df['Survival Time'][ix], event_observed=df['Year of death recode'][ix], label=str(group))
        kmf.plot_survival_function(ci_show=True)

    plt.title(f'Kaplan-Meier Survival Curve: {var}', fontsize=16)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Survival Probability', fontsize=14)

    # 调整图例的位置
    plt.legend(title=var, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize='13')

    # 显示网格
    plt.grid(True)

    # 添加注释
    # plt.annotate('This is a Kaplan-Meier Survival Curve', xy=(0.5, 0.1), xycoords='axes fraction',
    #              fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # 处理文件名中的特殊字符
    safe_var = sanitize_filename(var)
    plt.savefig(f'KM_Survival_Curve_{safe_var}.png', bbox_inches='tight')
    plt.close()

print("生存曲线已生成并保存。")
