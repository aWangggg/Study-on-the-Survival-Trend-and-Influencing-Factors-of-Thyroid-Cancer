import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib
import re
from sklearn.ensemble import RandomForestClassifier
# 使用非GUI后端
matplotlib.use('Agg')

# 读取数据
data = pd.read_excel('ThyroidCancer.xlsx')

# 查看数据缺失情况
print("数据缺失情况：")
print(data.isnull().sum())

# 定义特征和标签
feature_columns = [
    'Age', 'Sex', 'Race recode (W, B, AI, API)', 'Grade Pathological (2018+)',
    'RX Summ--Surg/Rad Seq', 'Radiation recode', 'Chemotherapy recode (yes, no/unk)',
    'Tumor Size Over Time Recode (1988+)', 'Tumor Size Summary (2016+)',
    'Regional nodes examined (1988+)', 'Regional nodes positive (1988+)',
    'SEER Combined Mets at DX-bone (2010+)', 'SEER Combined Mets at DX-brain (2010+)',
    'SEER Combined Mets at DX-liver (2010+)', 'SEER Combined Mets at DX-lung (2010+)',
    'CS tumor size (2004-2015)', 'CS extension (2004-2015)', 'Marital status at diagnosis'
]

# 过滤缺失值过多的列，阈值设置为50%
threshold = len(data) * 0.5
filtered_feature_columns = [col for col in feature_columns if data[col].isnull().sum() <= threshold]

print(f"过滤后的特征列：{filtered_feature_columns}")

# 删除缺失值过多的列
data = data[filtered_feature_columns + ['Survival Time']]

# 删除剩余的缺失值
data = data.dropna()

# 确认删除缺失值后数据集中仍有样本
if data.empty:
    raise ValueError("数据集中没有足够的样本进行训练和测试，请检查数据预处理步骤。")

# 提取特征和标签
X = data[filtered_feature_columns]
y = (data['Survival Time'] > 8).astype(int)  # 将生存时间超过8年作为标签

# 检查标签分布
print("标签分布：")
print(y.value_counts())

# 如果标签分布不均衡，进行重新采样
if y.value_counts().min() == 0:
    raise ValueError("数据集中没有包含足够的两个类别样本。请检查数据或选择其他特征列进行处理。")

# 将分类变量转换为数值
X = pd.get_dummies(X, drop_first=True)

# 确认处理后的特征和标签中仍有样本
if X.empty or y.empty:
    raise ValueError("在处理特征和标签时出现问题，请检查数据预处理步骤。")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 确认分割后的训练集和测试集中仍有样本
if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
    raise ValueError("在分割数据集时出现问题，请检查数据集大小和预处理步骤。")

# 定义模型
model = RandomForestClassifier(random_state=42)

# 进行K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')

print(f"K折交叉验证AUC得分: {cv_scores}")
print(f"AUC得分均值: {cv_scores.mean()}")

# 保存交叉验证结果
cv_results_df = pd.DataFrame(cv_scores, columns=['AUC Score'])
cv_results_df.to_csv('cross_val_results.csv', index=False)

# 训练模型并绘制ROC曲线
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 标出平衡点
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label=f'Optimal threshold = {optimal_threshold:.2f}')

# 标出多个阈值点
for i in range(0, len(thresholds), int(len(thresholds) / 10)):
    plt.scatter(fpr[i], tpr[i], marker='x', color='red')
    plt.text(fpr[i], tpr[i], f'Threshold = {thresholds[i]:.2f}\n(TPR={tpr[i]:.2f}, FPR={fpr[i]:.2f})', fontsize=8)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('ROC_Curve_result.png')
plt.close()

print("交叉验证结果已保存，ROC曲线已生成并保存。")
