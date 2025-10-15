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
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import seaborn as sns

warnings.filterwarnings('ignore')

#Read drug labels
root_dir='../RamanRawData'
classes=os.listdir(root_dir)
classes
L=[]
str1='x'
for i in classes:
    i=i[:i.index(str1)]
    i=i.rstrip('.')
    L.append(i)

#Load preprocessed data
summary=pd.read_excel("../SERS_Data_10%.xlsx")

melt=pd.melt(summary,id_vars=['potential','label','category'],value_name='intensity')
melt.dropna(how='any',inplace=True)

dfx=pd.DataFrame(columns=['potential','label','category','intensity'])
for i in range(xxcs):
    if i%100000==0:
        print('Progress：{:.2f}%'.format(i/xxcs*100))
    con=melt.iloc[[i],:]

    #Select the range of input data
    if con['label'].item()=='1.DNA_alkylation-Cyclophosphamide.xlsx' or con['label'].item()=='1.DNA_alkylation-carboplatin.xlsx' or con['label'].item()=='6.TopoisomeraseI_inhibition-Irinotecan.xlsx'or con['label'].item()=='6.TopoisomeraseI_inhibition-Topotecan.xlsx'or con['label'].item()=='7.HDAC_inhibition-Panobinostat.xlsx': 
        if con['potential'].item()>=840 and con['potential'].item()<=901:
            dfx=pd.concat([dfx,con],ignore_index=True)
            continue
    else:
        if con['potential'].item()>=840 and con['potential'].item()<=900:
            dfx=pd.concat([dfx,con],ignore_index=True)
            continue
    if con['potential'].item()>=1000 and con['potential'].item()<=1060:
        dfx=pd.concat([dfx,con],ignore_index=True)
        continue
    if con['potential'].item()>=1100 and con['potential'].item()<=1180:
        dfx=pd.concat([dfx,con],ignore_index=True)
        continue
       
dfx.dropna(how='any',inplace=True)
dfx.shape

category_dict={cat:index for index,cat in enumerate(dfx['category'].unique())}
category_dict
label_dict={lab:index for index,lab in enumerate(dfx['label'].unique())}
label_dict
dfx['category']=dfx['category'].map(category_dict)
dfx['label']=dfx['label'].map(label_dict)

#Build x, y
def generator_x(df):
    x=df[['potential','category','intensity']].values.T
    y=df['label'].values[0]
    return x

def generator_y(df):
    x=df[['potential','category','intensity']].values.T
    y=df['label'].values[0]
    return y
x=dfx.groupby(['label','category','variable'],group_keys=False).apply(generator_x)
y=dfx.groupby(['label','category','variable'],group_keys=False).apply(generator_y)

#Remove data that does not match the input size
D=[]
for i in range(len(x)):
    print(x.values[i].shape[1])
    if x.values[i].shape[1]!=178:
        D.append(x.index[i])
x.drop(D,inplace=True)
y.drop(D,inplace=True)

X=np.stack(x.values)
Y=y.values

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Printing device information
print(f"Using equipment: {device}")


#Convert NumPy arrays to PyTorch tensors
X = torch.from_numpy(X.astype(float)).float()  #Ensure that the data type is float
y = torch.from_numpy(Y).long()   #Tags usually need to be of the long type

#Divide the training set and testing set
seed=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

#Definition model
class SERS_VGG27(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        #1
        self.conv1=nn.Conv1d(3,64,kernel_size=1,padding=1)
        self.Bat1=nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        #2
        self.conv2=nn.Conv1d(64,64,kernel_size=3,padding=1)
        self.Bat2=nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        #3
        self.conv3=nn.Conv1d(64,64,kernel_size=1,padding=1)
        self.Bat3=nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        #4
        self.conv4=nn.Conv1d(64,64,kernel_size=3,padding=1)
        self.Bat4=nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        self.Mpol=nn.MaxPool1d(kernel_size=2,stride=2)
        
        #5
        self.conv5=nn.Conv1d(64,128,kernel_size=1,padding=1)
        self.Bat5=nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        #6
        self.conv6=nn.Conv1d(128,128,kernel_size=3,padding=1)
        self.Bat6=nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        #7
        self.conv7=nn.Conv1d(128,128,kernel_size=1,padding=1)
        self.Bat7=nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)        
        #8
        self.conv8=nn.Conv1d(128,128,kernel_size=3,padding=1)
        self.Bat8=nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.Mpo2=nn.MaxPool1d(kernel_size=2,stride=2)
        
        #9
        self.conv9=nn.Conv1d(128,256,kernel_size=1,padding=1)
        self.Bat9=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        #10
        self.conv10=nn.Conv1d(256,256,kernel_size=3,padding=1)
        self.Bat10=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        #11
        self.conv11=nn.Conv1d(256,256,kernel_size=1,padding=1)
        self.Bat11=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        #12
        self.conv12=nn.Conv1d(256,256,kernel_size=3,padding=1)
        self.Bat12=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        self.Mpo3=nn.MaxPool1d(kernel_size=2,stride=2)
        
         #13
        self.conv13=nn.Conv1d(256,256,kernel_size=1,padding=1)
        self.Bat13=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        #14
        self.conv14=nn.Conv1d(256,256,kernel_size=3,padding=1)
        self.Bat14=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)      
        #15
        self.conv15=nn.Conv1d(256,256,kernel_size=1,padding=1)
        self.Bat15=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        #16
        self.conv16=nn.Conv1d(256,256,kernel_size=3,padding=1)
        self.Bat16=nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        self.Mpo4=nn.MaxPool1d(kernel_size=2,stride=2)
        
        #17
        self.conv17=nn.Conv1d(256,512,kernel_size=1,padding=1)
        self.Bat17=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        #18
        self.conv18=nn.Conv1d(512,512,kernel_size=3,padding=1)
        self.Bat18=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)        
        #19
        self.conv19=nn.Conv1d(512,512,kernel_size=1,padding=1)
        self.Bat19=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        #20
        self.conv20=nn.Conv1d(512,512,kernel_size=3,padding=1)
        self.Bat20=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        self.Mpo5=nn.MaxPool1d(kernel_size=2,stride=2)
        
        #21
        self.conv21=nn.Conv1d(512,512,kernel_size=1,padding=1)
        self.Bat21=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        #22
        self.conv22=nn.Conv1d(512,512,kernel_size=3,padding=1)
        self.Bat22=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)       
       #23
        self.conv23=nn.Conv1d(512,512,kernel_size=1,padding=1)
        self.Bat23=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
       #24
        self.conv24=nn.Conv1d(512,512,kernel_size=3,padding=1)
        self.Bat24=nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        self.Mpo6=nn.MaxPool1d(kernel_size=2,stride=2)
        
        self.pol2=nn.AvgPool1d(kernel_size=1,stride=1)

        self.classifier = nn.Sequential(
            # 25
            nn.Linear(3072, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.2),
            # 26
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.2),
            # 27
            nn.Linear(4096, len(label_dict)),
        )

        

       
 
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.Bat1(out)
        out = self.relu(out)       
        
        out = self.conv2(out)
        out = self.Bat2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.Bat3(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.Bat4(out)
        out = self.relu(out)
        out = self.Mpol(out)


        out = self.conv5(out)
        out = self.Bat5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.Bat6(out)
        out = self.relu(out)
        
        out = self.conv7(out)
        out = self.Bat7(out)
        out = self.relu(out)
        
        out = self.conv8(out)        
        out = self.Bat8(out)
        out = self.relu(out)
        out = self.Mpo2(out)
      
        out = self.conv9(out)
        out = self.Bat9(out)
        out = self.relu(out)

        out = self.conv10(out)
        out = self.Bat10(out)
        out = self.relu(out)
        
        out = self.conv11(out)
        out = self.Bat11(out)
        out = self.relu(out)
        
        out = self.conv12(out)
        out = self.Bat12(out)
        out = self.relu(out)
        out = self.Mpo3(out)
        
        out = self.conv13(out)
        out = self.Bat13(out)
        out = self.relu(out)
        
        out = self.conv14(out)
        out = self.Bat14(out)
        out = self.relu(out)
        
        out = self.conv15(out)
        out = self.Bat15(out)
        out = self.relu(out)
        
        out = self.conv16(out)
        out = self.Bat16(out)
        out = self.relu(out)
        out = self.Mpo4(out)

        out = self.conv17(out)
        out = self.Bat17(out)
        out = self.relu(out)
        
        out = self.conv18(out)
        out = self.Bat18(out)
        out = self.relu(out)
        
        out = self.conv19(out)
        out = self.Bat19(out)
        out = self.relu(out)
        
        out = self.conv20(out)
        out = self.Bat20(out)
        out = self.relu(out)
        out = self.Mpo5(out)

        out = self.conv21(out)
        out = self.Bat21(out)
        out = self.relu(out)
        
        out = self.conv22(out)
        out = self.Bat22(out)
        out = self.relu(out)
        
        out = self.conv23(out)
        out = self.Bat23(out)
        out = self.relu(out)
        
        out = self.conv24(out)
        out = self.Bat24(out)
        out = self.relu(out)
        out = self.Mpo6(out)
        
        out = self.pol2(out)
        
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out

#Define training and testing functions
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #Prediction&Calculation Error
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn,name):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"{name} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct,test_loss

#Train the model
epochs = 200
batch_size=128
learning_rate=0.0001

#Fixed random seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model = SERS_VGG27().to(device)


#Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

history=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, criterion, optimizer)
    train_accuracy,train_loss=test_loop(train_loader, model, criterion,'Train')
    test_accuracy,test_loss=test_loop(test_loader, model, criterion,'Test')
    history.append([train_accuracy,train_loss,test_accuracy,test_loss])
print("Done!")

#Evaluation model
#Visualization of Training Process
sns.set_theme()
fig, ax=plt.subplots(1,2,figsize=[16,5])
history.loc[:,['train_accuracy','test_accuracy']].plot(ax=ax[0])
ax[0].set_ylabel('Accuracy')
history.loc[:,['train_loss','test_loss']].plot(ax=ax[1])
ax[1].set_ylabel('Cross Entropy Loss')
fig.suptitle('Learning Curve')
plt.show()

with torch.no_grad():
    predict=[]
    target=[]
    y_score=[]
    s_score=[]
    score=[]
    for feature, label in test_loader:
        pred = model(feature.to(device)).argmax(1)
        for x in model(feature.to(device)).cpu().numpy():
            y_score.append(x)
        predict.append(pred.cpu().numpy())
        target.append(label.cpu().numpy())
    predict=np.hstack(predict)
    target=np.hstack(target)

#Normalization function
def maxminnorm(array):
    max=array.max()
    min=array.min()
    t=[]
    for i in array:
        t.append((i-min)/(max-min))
    return t

for y in y_score:
    s_score=[]
    sum=0
    y=maxminnorm(y)
    for k in y:
        sum=sum+k
    for k in y:
        k=k/sum
        s_score.append(k)
    score.append(s_score)

y_score=[]
for i in range(21):
    f=[]
    for y in score:
        f.append(y[i])
    y_score.append(f)

acc=int(history['test_accuracy'].values[-1]*100)
torch.save(model, f'model_{acc}.pth')#Save model
model = torch.load(f'model_{acc}.pth')#Load model

#Output confusion matrix
fig,ax=plt.subplots(figsize=[10,8])
ConfusionMatrixDisplay.from_predictions(target,predict,ax=ax,cmap='Blues')
ax.grid(False)
ax.set_title("ConfusionMatrixDisplay Accuracy:{:.2f}%".format(history['test_accuracy'].values[-1]*100),fontsize=16,pad=20)
plt.show()

#Output ROC curve
fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(21):
    fpr[i],tpr[i],_=metrics.roc_curve(y_true=target,y_score=y_score[i],pos_label=i)
    roc_auc[i]=metrics.auc(fpr[i],tpr[i])

plt.figure(figsize=(15,10))
colors=['black','dimgray','silver','lightcoral','brown',
        'red','tomato','orangered','darkred','sienna',
        'peachpuff','darkorange','tan','gold','yellow',
       'yellowgreen','palegreen','green','cyan','lightskyblue','dodgerblue']
for i,color in zip(range(21),colors) :
    plt.plot(fpr[i],tpr[i],color=color,lw=2,label='ROC curve of {0} (area={1:0.2f})'
             ''.format(L[i],roc_auc[i]))
plt.plot([0,1],[0,1],'k--',lw=2)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.legend(loc='lower right')
plt.show()
