import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy
import re
import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

# ==========================================
# 1. Environment Setup & Data Loading
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current computing device: {device}")

# Load pre-processed data
npz_path = "Drug-SERS Fine-Grained Feature Dataset_Z-score Standardization.npz"
print(f"Loading processed data from {npz_path}...")
data = np.load(npz_path, allow_pickle=True)

X_final = data['X']
y_final = data['y']
classes_ = data['classes']
print(f"Data loaded successfully! Tensor shape: {X_final.shape}, Number of classes: {len(classes_)}")

num_classes = len(classes_)

# Mock label encoder for visualization compatibility
class MockLabelEncoder:
    def __init__(self, classes):
        self.classes_ = classes
le_label = MockLabelEncoder(classes_)

# Extract categories for stratified sampling
cat_final = X_final[:, 1, 0]  
label_final = y_final


# ==========================================
# 2. Model Definition (SERS_VGG27)
# ==========================================
class SERS_VGG27(nn.Module):
    def __init__(self, num_classes=1000):
        super(SERS_VGG27, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, padding=1)
        self.Bat1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.Bat2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, padding=1)
        self.Bat3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.Bat4 = nn.BatchNorm1d(64)
        self.Mpol = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv5 = nn.Conv1d(64, 128, kernel_size=1, padding=1)
        self.Bat5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.Bat6 = nn.BatchNorm1d(128)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=1, padding=1)
        self.Bat7 = nn.BatchNorm1d(128)        
        self.conv8 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.Bat8 = nn.BatchNorm1d(128)
        self.Mpo2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv9 = nn.Conv1d(128, 256, kernel_size=1, padding=1)
        self.Bat9 = nn.BatchNorm1d(256)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat10 = nn.BatchNorm1d(256)
        self.conv11 = nn.Conv1d(256, 256, kernel_size=1, padding=1)
        self.Bat11 = nn.BatchNorm1d(256)
        self.conv12 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat12 = nn.BatchNorm1d(256)
        self.Mpo3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv13 = nn.Conv1d(256, 256, kernel_size=1, padding=1)
        self.Bat13 = nn.BatchNorm1d(256)
        self.conv14 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat14 = nn.BatchNorm1d(256)      
        self.conv15 = nn.Conv1d(256, 256, kernel_size=1, padding=1)
        self.Bat15 = nn.BatchNorm1d(256)
        self.conv16 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.Bat16 = nn.BatchNorm1d(256)
        self.Mpo4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv17 = nn.Conv1d(256, 512, kernel_size=1, padding=1)
        self.Bat17 = nn.BatchNorm1d(512)
        self.conv18 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat18 = nn.BatchNorm1d(512)        
        self.conv19 = nn.Conv1d(512, 512, kernel_size=1, padding=1)
        self.Bat19 = nn.BatchNorm1d(512)
        self.conv20 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat20 = nn.BatchNorm1d(512)
        self.Mpo5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 6
        self.conv21 = nn.Conv1d(512, 512, kernel_size=1, padding=1)
        self.Bat21 = nn.BatchNorm1d(512)
        self.conv22 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat22 = nn.BatchNorm1d(512)        
        self.conv23 = nn.Conv1d(512, 512, kernel_size=1, padding=1)
        self.Bat23 = nn.BatchNorm1d(512)
        self.conv24 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.Bat24 = nn.BatchNorm1d(512)
        self.Mpo6 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(7) 

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.Mpol(self.relu(self.Bat4(self.conv4(self.relu(self.Bat3(self.conv3(self.relu(self.Bat2(self.conv2(self.relu(self.Bat1(self.conv1(x)))))))))))))
        out = self.Mpo2(self.relu(self.Bat8(self.conv8(self.relu(self.Bat7(self.conv7(self.relu(self.Bat6(self.conv6(self.relu(self.Bat5(self.conv5(out)))))))))))))
        out = self.Mpo3(self.relu(self.Bat12(self.conv12(self.relu(self.Bat11(self.conv11(self.relu(self.Bat10(self.conv10(self.relu(self.Bat9(self.conv9(out)))))))))))))
        out = self.Mpo4(self.relu(self.Bat16(self.conv16(self.relu(self.Bat15(self.conv15(self.relu(self.Bat14(self.conv14(self.relu(self.Bat13(self.conv13(out)))))))))))))
        out = self.Mpo5(self.relu(self.Bat20(self.conv20(self.relu(self.Bat19(self.conv19(self.relu(self.Bat18(self.conv18(self.relu(self.Bat17(self.conv17(out)))))))))))))
        out = self.Mpo6(self.relu(self.Bat24(self.conv24(self.relu(self.Bat23(self.conv23(self.relu(self.Bat22(self.conv22(self.relu(self.Bat21(self.conv21(out)))))))))))))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# ==========================================
# 3. 5-Fold Cross Validation & Training
# ==========================================
n_splits = 5        
num_chunks = 20     

group_indices = {}
for idx, (lbl, cat) in enumerate(zip(label_final, cat_final)):
    key = (lbl, cat)
    if key not in group_indices:
        group_indices[key] = []
    group_indices[key].append(idx)

# Data splitting logic (Train: 12, Val: 4, Test: 4 chunks)
folds_train_indices = [[] for _ in range(n_splits)]
folds_val_indices = [[] for _ in range(n_splits)]  
folds_test_indices = [[] for _ in range(n_splits)]

for key, indices in group_indices.items():
    chunks = np.array_split(indices, num_chunks)
    chunk_indices_shuffled = np.random.permutation(num_chunks)
    
    for fold in range(n_splits):
        test_chunk_idx = chunk_indices_shuffled[fold * 4 : (fold + 1) * 4]
        remaining_chunk_idx = [i for i in range(num_chunks) if i not in test_chunk_idx]
        val_chunk_idx = np.random.choice(remaining_chunk_idx, size=4, replace=False)
        train_chunk_idx = [i for i in remaining_chunk_idx if i not in val_chunk_idx]
        
        for idx in test_chunk_idx:
            folds_test_indices[fold].extend(chunks[idx])
        for idx in val_chunk_idx:
            folds_val_indices[fold].extend(chunks[idx])
        for idx in train_chunk_idx:
            folds_train_indices[fold].extend(chunks[idx])

y_true_all = []
y_pred_all = []
y_score_all = []
fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

BATCH_SIZE = 32
EPOCHS = 200  
LEARNING_RATE = 0.0001
PATIENCE = 40  

print(f"\nStarting {n_splits}-fold cross-validation (SERS_VGG27)...")

for fold in range(n_splits):
    train_index = np.array(folds_train_indices[fold])
    val_index = np.array(folds_val_indices[fold])      
    test_index = np.array(folds_test_indices[fold])
    
    print(f"\n>>> Fold {fold+1}/{n_splits} Processing...")
    
    X_train, y_train = X_final[train_index], y_final[train_index]
    X_val, y_val = X_final[val_index], y_final[val_index]
    X_test, y_test = X_final[test_index], y_final[test_index]
    
    tensor_x_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    tensor_x_val = torch.tensor(X_val, dtype=torch.float32)
    tensor_y_val = torch.tensor(y_val, dtype=torch.long)
    tensor_x_test = torch.tensor(X_test, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(tensor_x_val, tensor_y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(tensor_x_test, tensor_y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = SERS_VGG27(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0  
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}. Best Val Acc: {best_acc:.4f}")
                break
    
    model.load_state_dict(best_model_wts)
            
    model.eval()
    fold_probs = []
    fold_preds = []
    fold_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            fold_probs.append(probs.cpu().numpy())
            fold_preds.append(preds.cpu().numpy())
            fold_true.append(labels.numpy())
            
    y_test_fold = np.concatenate(fold_true)
    y_pred_fold = np.concatenate(fold_preds)
    y_prob_fold = np.concatenate(fold_probs)
    
    acc = accuracy_score(y_test_fold, y_pred_fold)
    prec = precision_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
    rec = recall_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
    f1 = f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
    
    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)
    
    print(f"  Fold {fold+1} Final Test Metrics: Acc: {acc:.2%}, Prec: {prec:.2%}, Rec: {rec:.2%}, F1: {f1:.2%}")
    
    y_true_all.extend(y_test_fold)
    y_pred_all.extend(y_pred_fold)
    y_score_all.append(y_prob_fold)
    
    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_score_all = np.concatenate(y_score_all, axis=0)

print("\n==========================================")
print("5-Fold Cross Validation Results on Independent Test Sets (Macro Avg ± Std):")
print(f"Accuracy:  {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
print(f"Precision: {np.mean(fold_precisions)*100:.2f}% ± {np.std(fold_precisions)*100:.2f}%")
print(f"Recall:    {np.mean(fold_recalls)*100:.2f}% ± {np.std(fold_recalls)*100:.2f}%")
print(f"F1-Score:  {np.mean(fold_f1s)*100:.2f}% ± {np.std(fold_f1s)*100:.2f}%")
print("==========================================\n")

print("Cross-validation complete. Data converted to Numpy arrays for visualization.")

# ==========================================
# 4. Results Visualization
# ==========================================
# Ensure data is in Numpy format
if isinstance(y_true_all, list):
    y_true_all = np.array(y_true_all)
if isinstance(y_pred_all, list):
    y_pred_all = np.array(y_pred_all)
if isinstance(y_score_all, list):
    y_score_all = np.concatenate(y_score_all, axis=0)

# Extract class names and remove '.xlsx' suffix
raw_names = list(le_label.classes_)
target_names = [str(name).replace('.xlsx', '') for name in raw_names]

overall_acc = metrics.accuracy_score(y_true_all, y_pred_all)
print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, 
                            target_names=target_names, 
                            digits=4,
                            zero_division=0))

# Plot Style Configuration
plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=[14, 12])
cmd = ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_all, 
                                        display_labels=target_names,
                                        ax=ax, cmap='Blues', 
                                        normalize=None, 
                                        values_format='d',
                                        colorbar=False) 

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(cmd.im_, cax=cax)

ax.grid(False)
ax.set_title(f"Confusion Matrix Accuracy: {overall_acc*100:.2f}%", fontsize=25, pad=20)
ax.set_xlabel("Predicted Label")
ax.set_xticklabels([]) 
ax.set_yticks(range(len(target_names)))
ax.set_yticklabels(target_names, rotation=0)

ylabels = ax.get_yticklabels()
for label in ylabels:
    label.set_ha('left')
    label.set_position((-0.5, 0)) 
    label.set_fontsize(16) 

plt.tight_layout()
plt.show()

# Plot ROC Curves
plt.rcParams.update({'legend.fontsize': 14})
grouped_indices = {i: [] for i in range(1, 10)} 

for idx, name in enumerate(target_names):
    match = re.match(r"(\d+)\.", str(name))
    if match:
        group_num = int(match.group(1))
        if group_num in grouped_indices:
            grouped_indices[group_num].append(idx)
        else:
            grouped_indices[9].append(idx)
    else:
        grouped_indices[9].append(idx)

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
        label_name = target_names[i]
        plt.plot(fpr, tpr, lw=2, label='{0} (area={1:0.4f})'.format(label_name, roc_auc_val))
        found_any = True

    if found_any:
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        title_suffix = f" - Group {group_id}" if group_id < 9 else " - All Classes"
        plt.title(f'ROC Curves{title_suffix}')
        plt.legend(loc="lower right", fontsize=17) 
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
