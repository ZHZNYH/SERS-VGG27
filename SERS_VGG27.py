import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn import metrics
import re

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 0.Set Matplotlib Chinese font to prevent garbled graphics
# -----------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  

# -----------------------------------------------------------------------------
# 1. Data Reading and Label Processing
# -----------------------------------------------------------------------------

# 请确保路径正确
root_dir = 'C:\\Users\\HS\\Desktop\\文件夹\\论文数据\\拉曼\\原始拉曼数据_英文名称'
data_path = "C:\\Users\\HS\\Desktop\\文件夹\\论文数据\\拉曼\\清洗后的拉曼数据\\SERS_Data_10%.xlsx"

# --- 步骤A: (仅供参考) 读取文件夹下的英文名称并打印 ---
print("="*50)
print("【参考信息】原始文件夹下的英文文件名称列表：")
if os.path.exists(root_dir):
    raw_files = os.listdir(root_dir)
    english_labels_ref = []
    
    for i in raw_files:
        if i.endswith('.xlsx') or i.endswith('.csv') or i.endswith('.txt'):
            name = os.path.splitext(i)[0] 
            english_labels_ref.append(name)
        elif os.path.isdir(os.path.join(root_dir, i)):
            english_labels_ref.append(i)
    
    # 简单的排序显示
    def extract_prefix_num(s):
        match = re.match(r"(\d+)\.", s)
        return int(match.group(1)) if match else 999
    
    english_labels_ref = sorted(english_labels_ref, key=extract_prefix_num)
    for idx, name in enumerate(english_labels_ref):
        print(f"  {idx+1}. {name}")
else:
    print(f"Warning: {root_dir} not found. 无法列出英文参考名称。")
print("="*50)


# --- 步骤B: 读取Excel数据并清洗 ---
if not os.path.exists(data_path):
    raise FileNotFoundError(f"找不到文件: {data_path}")

print(f"正在读取数据: {data_path}")
summary = pd.read_excel(data_path)
melt = pd.melt(summary, id_vars=['potential', 'label', 'category'], value_name='intensity')
melt.dropna(how='any', inplace=True)

print("开始数据筛选...")
# 筛选逻辑 (保持不变)
special_labels = [
    '1.DNA_alkylation-Cyclophosphamide.xlsx', '1.DNA_alkylation-carboplatin.xlsx',
    '6.TopoisomeraseI_inhibition-Irinotecan.xlsx', '6.TopoisomeraseI_inhibition-Topotecan.xlsx',
    '7.HDAC_inhibition-Panobinostat.xlsx'
]

# 注意：这里的筛选依赖于 label 列的内容，请确保 Excel 中的 label 列内容符合筛选条件
mask_special = melt['label'].isin(special_labels) & (melt['potential'] >= 840) & (melt['potential'] <= 901)
mask_normal = (~melt['label'].isin(special_labels)) & (melt['potential'] >= 840) & (melt['potential'] <= 900)
mask_common1 = (melt['potential'] >= 1000) & (melt['potential'] <= 1060)
mask_common2 = (melt['potential'] >= 1100) & (melt['potential'] <= 1180)

dfx = melt[mask_special | mask_normal | mask_common1 | mask_common2].copy()
dfx.dropna(how='any', inplace=True)
print(f"筛选完成，数据形状: {dfx.shape}")

# --- 步骤C: 直接使用中文标签 ---

# 1. 获取 Excel 中所有的唯一中文标签
chinese_labels_raw = dfx['label'].unique()

# 2. 定义排序逻辑 (提取 "1.", "2." 等前缀进行排序)
def sort_chinese_label(s):
    # 将输入转为字符串，防止读取为其他类型
    s_str = str(s)
    match = re.match(r"(\d+)\.", s_str)
    if match:
        return int(match.group(1))
    else:
        return 999 # 如果没有数字前缀，排在最后

# 3. 对中文标签进行排序
sorted_chinese_labels = sorted(chinese_labels_raw, key=sort_chinese_label)

# 4. 建立映射字典 {中文名: ID}
label_dict = {label: idx for idx, label in enumerate(sorted_chinese_labels)}

# 5. 设置绘图用的目标名称为中文列表
target_names = [str(l) for l in sorted_chinese_labels]

print("\n【确认】最终用于训练和显示的标签（中文）：")
for idx, name in enumerate(target_names):
    print(f"  ID {idx}: {name}")

# 处理 category
category_dict = {cat: index for index, cat in enumerate(dfx['category'].unique())}

dfx['category'] = dfx['category'].map(category_dict)
dfx['label'] = dfx['label'].map(label_dict)

# 生成数据矩阵
def generator_x(df):
    return df[['potential', 'category', 'intensity']].values.T

def generator_y(df):
    return df['label'].values[0]

print("正在生成张量数据...")
x = dfx.groupby(['label', 'category', 'variable'], group_keys=False).apply(generator_x)
y = dfx.groupby(['label', 'category', 'variable'], group_keys=False).apply(generator_y)

# -----------------------------------------------------------------------------
# [预处理] 数据长度调整至 178
# -----------------------------------------------------------------------------
def adjust_length(data, target_len=178):
    c, n = data.shape
    if n == target_len:
        return data
    if n > target_len:
        start = (n - target_len) // 2
        end = start + target_len
        return data[:, start:end]
    else: 
        diff = target_len - n
        pad_left = diff // 2
        pad_right = diff - pad_left
        return np.pad(data, ((0, 0), (pad_left, pad_right)), mode='edge')

print(f"正在标准化数据长度至 178 (样本总数: {len(x)})...")
X_list = []
Y_list = []

for i in range(len(x)):
    raw_data = x.iloc[i] 
    label_id = y.iloc[i]
    processed_data = adjust_length(raw_data, target_len=178)
    X_list.append(processed_data)
    Y_list.append(label_id)

X_raw = np.stack(X_list)
Y_raw = np.array(Y_list)
print(f"数据标准化完成。最终输入形状: {X_raw.shape}")

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using equipment: {device}")

# 转换为 Numpy 数组
X_numpy = np.array(X_raw, dtype=np.float32)
y_numpy = np.array(Y_raw, dtype=np.int64)

# 转换为 Tensor
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
y_tensor = torch.tensor(y_numpy, dtype=torch.long)

# -----------------------------------------------------------------------------
# 2. 模型定义 (SERS_VGG27) - 保持不变
# -----------------------------------------------------------------------------

class SERS_VGG27(nn.Module):
    def __init__(self, num_classes):
        super(SERS_VGG27, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, padding=1)
        self.Bat1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        # Layer 2
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.Bat2 = nn.BatchNorm1d(64)
        # Layer 3
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, padding=1)
        self.Bat3 = nn.BatchNorm1d(64)
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.Bat4 = nn.BatchNorm1d(64)
        self.Mpol = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 5-8
        self.conv5 = nn.Conv1d(64, 128, kernel_size=1, padding=1)
        self.Bat5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.Bat6 = nn.BatchNorm1d(128)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=1, padding=1)
        self.Bat7 = nn.BatchNorm1d(128)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.Bat8 = nn.BatchNorm1d(128)
        self.Mpo2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 9-12
        self.conv9 = nn.Conv1d(128, 256, kernel_size=1, padding=1)
        self.Bat9 = nn.BatchNorm1d(256)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat10 = nn.BatchNorm1d(256)
        self.conv11 = nn.Conv1d(256, 256, kernel_size=1, padding=1)
        self.Bat11 = nn.BatchNorm1d(256)
        self.conv12 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat12 = nn.BatchNorm1d(256)
        self.Mpo3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 13-16
        self.conv13 = nn.Conv1d(256, 256, kernel_size=1, padding=1)
        self.Bat13 = nn.BatchNorm1d(256)
        self.conv14 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat14 = nn.BatchNorm1d(256)
        self.conv15 = nn.Conv1d(256, 256, kernel_size=1, padding=1)
        self.Bat15 = nn.BatchNorm1d(256)
        self.conv16 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat16 = nn.BatchNorm1d(256)
        self.Mpo4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 17-20
        self.conv17 = nn.Conv1d(256, 512, kernel_size=1, padding=1)
        self.Bat17 = nn.BatchNorm1d(512)
        self.conv18 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat18 = nn.BatchNorm1d(512)
        self.conv19 = nn.Conv1d(512, 512, kernel_size=1, padding=1)
        self.Bat19 = nn.BatchNorm1d(512)
        self.conv20 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat20 = nn.BatchNorm1d(512)
        self.Mpo5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 21-24
        self.conv21 = nn.Conv1d(512, 512, kernel_size=1, padding=1)
        self.Bat21 = nn.BatchNorm1d(512)
        self.conv22 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat22 = nn.BatchNorm1d(512)
        self.conv23 = nn.Conv1d(512, 512, kernel_size=1, padding=1)
        self.Bat23 = nn.BatchNorm1d(512)
        self.conv24 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat24 = nn.BatchNorm1d(512)
        self.Mpo6 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.pol2 = nn.AvgPool1d(kernel_size=1, stride=1)

        self.classifier = nn.Sequential(
            nn.Linear(3072, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.relu(self.Bat1(self.conv1(x)))
        out = self.relu(self.Bat2(self.conv2(out)))
        out = self.relu(self.Bat3(self.conv3(out)))
        out = self.Mpol(self.relu(self.Bat4(self.conv4(out))))

        out = self.relu(self.Bat5(self.conv5(out)))
        out = self.relu(self.Bat6(self.conv6(out)))
        out = self.relu(self.Bat7(self.conv7(out)))
        out = self.Mpo2(self.relu(self.Bat8(self.conv8(out))))
      
        out = self.relu(self.Bat9(self.conv9(out)))
        out = self.relu(self.Bat10(self.conv10(out)))
        out = self.relu(self.Bat11(self.conv11(out)))
        out = self.Mpo3(self.relu(self.Bat12(self.conv12(out))))
        
        out = self.relu(self.Bat13(self.conv13(out)))
        out = self.relu(self.Bat14(self.conv14(out)))
        out = self.relu(self.Bat15(self.conv15(out)))
        out = self.Mpo4(self.relu(self.Bat16(self.conv16(out))))

        out = self.relu(self.Bat17(self.conv17(out)))
        out = self.relu(self.Bat18(self.conv18(out)))
        out = self.relu(self.Bat19(self.conv19(out)))
        out = self.Mpo5(self.relu(self.Bat20(self.conv20(out))))

        out = self.relu(self.Bat21(self.conv21(out)))
        out = self.relu(self.Bat22(self.conv22(out)))
        out = self.relu(self.Bat23(self.conv23(out)))
        out = self.Mpo6(self.relu(self.Bat24(self.conv24(out))))
        
        out = self.pol2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# -----------------------------------------------------------------------------
# 3. 10折交叉验证
# -----------------------------------------------------------------------------

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = [] 
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch_y.tolist()) 
            all_probs.extend(probs.cpu().tolist())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

# 参数设置
n_splits = 10
epochs = 200
batch_size = 128
learning_rate = 0.0001
seed = 42
num_classes = len(label_dict)

cv_targets = []
cv_preds = []
cv_probs = []

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

print(f"\nStarting {n_splits}-Fold Cross-Validation...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_numpy, y_numpy)):
    print(f"Fold {fold+1}/{n_splits}")
    
    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
    test_idx_tensor = torch.tensor(test_idx, dtype=torch.long)
    
    X_train_fold = X_tensor[train_idx_tensor]
    X_test_fold = X_tensor[test_idx_tensor]
    
    y_train_fold = y_tensor[train_idx_tensor]
    y_test_fold = y_tensor[test_idx_tensor]
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    test_dataset = TensorDataset(X_test_fold, y_test_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SERS_VGG27(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer)
        if (epoch+1) % 50 == 0:
             print(f"  Epoch {epoch+1}/{epochs}")
            
    fold_targets, fold_preds, fold_probs = evaluate_model(model, test_loader)
    fold_acc = metrics.accuracy_score(fold_targets, fold_preds)
    print(f"  Fold {fold+1} Accuracy: {fold_acc*100:.2f}%")
    
    cv_targets.append(fold_targets)
    cv_preds.append(fold_preds)
    cv_probs.append(fold_probs)

# -----------------------------------------------------------------------------
# 4. 结果汇总与可视化
# -----------------------------------------------------------------------------

y_true_all = np.concatenate(cv_targets)
y_pred_all = np.concatenate(cv_preds)
y_score_all = np.concatenate(cv_probs)

overall_acc = metrics.accuracy_score(y_true_all, y_pred_all)
print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")

print("\nClassification Report (Chinese Labels):")
# 直接使用中文 target_names
print(classification_report(y_true_all, y_pred_all, 
                            target_names=target_names, 
                            digits=4,
                            zero_division=0))

# --- 混淆矩阵 ---
fig, ax = plt.subplots(figsize=[16, 14]) # 稍微调大图片尺寸以容纳中文
cmd = ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_all, 
                                        display_labels=target_names, # 显示中文
                                        ax=ax, cmap='Blues', 
                                        normalize=None, 
                                        values_format='d')

ax.grid(False)
ax.set_title(f"Confusion Matrix (Counts) - Acc: {overall_acc*100:.2f}%", fontsize=16, pad=20)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
# X轴标签旋转，防止中文重叠
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- ROC 曲线 (按数字前缀分组 1-9) ---
grouped_indices = {i: [] for i in range(1, 10)} 

for idx, name in enumerate(target_names):
    # 提取中文名称中的数字前缀 "1."
    match = re.match(r"(\d+)\.", str(name))
    if match:
        group_num = int(match.group(1))
        if group_num in grouped_indices:
            grouped_indices[group_num].append(idx)
        else:
            grouped_indices[9].append(idx)
    else:
        grouped_indices[9].append(idx)

# 循环绘图
for group_id in range(1, 10):
    indices = grouped_indices[group_id]
    if not indices:
        continue 
        
    plt.figure(figsize=(10, 8))
    
    found_any = False
    for i in indices:
        if np.sum(y_true_all == i) == 0:
            continue
            
        fpr, tpr, _ = metrics.roc_curve(y_true_all == i, y_score_all[:, i])
        roc_auc_val = metrics.auc(fpr, tpr)
        
        # 这里的 label_name 是中文
        label_name = target_names[i]
        plt.plot(fpr, tpr, lw=2,
                 label='{0} (area={1:0.4f})'.format(label_name, roc_auc_val))
        found_any = True

    if found_any:
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - Group {group_id}')
        plt.legend(loc="lower right") # 图例中会显示中文
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
