import os
import pandas
import torch
# -*- coding: utf-8 -*-
# pytorch mlp for multiclass classification
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Softmax, Module, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# come from https://blog.csdn.net/jclian91/article/details/121708431
# dataset definition

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=0) # header=0 参数使 pandas 使用 CSV 文件的第一行作为列名，从而不会把它作为数据的一部分。header=None
        # store the inputs and outputs
        self.X = df.values[:, [3, 4, 5, 7]]
        self.y = df.values[:, -2]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y) # 使用 LabelEncoder 对标签列进行编码。
        # self.y = LabelBinarizer().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
  

# model definition
class MLP(Module):  # 实例化 MLP 时指定Module n_inputs
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 16)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(16, 14)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(14, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        # self.act3 = Softmax(dim=1) # No softmax here as CrossEntropyLoss expects raw logits
        # 4
        self.hidden4 = Linear(10, 14)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act4 = ReLU()
        # 5
        self.hidden5 = Linear(14, 6)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # 6
        self.hidden6 = Linear(6, 5)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act6 = ReLU()
        # 7
        self.hidden7 = Linear(5, 5)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act7 = Softmax(dim=1)
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        # X = self.act3(X) Softmax 被移除，因为 CrossEntropyLoss 会在内部处理 logits 和 softmax 计算
        X = self.hidden4(X)
        X = self.act4(X)
        # 5th layer
        X = self.hidden5(X)
        X = self.act5(X)
        # 6
        X = self.hidden6(X)
        X = self.act6(X)
        # 7
        X = self.hidden7(X)
        X = self.act7(X)
        return X


# prepare the dataset
def prepare_data(train_path, test_path):
    # load the dataset
    train = CSVDataset(train_path)
    test = CSVDataset(test_path)
    # calculate split
    # train, test = dataset.get_splits() 已经自动划分

    # prepare data loaders
    train_dl = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model, epoch):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    log = {
        "Model Structure": str(model),
        "Train Batch Size": train_dl.batch_size,
        "Test Batch Size": test_dl.batch_size,
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Epochs": epoch
    }
    logs_epoch = []

    # optimizer = Adam(model.parameters())
    # enumerate epochs
    for epoch in range(epoch):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.long()
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            s = "epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data)
            print(s)
            # update model weights
            optimizer.step()
            log_entry = {
                'epoch': epoch,
                'batch': i,
                'loss': loss.item()  # 使用 item() 获取损失值的标量
            }
            logs_epoch.append(log_entry)
    # 将日志信息转换为 DataFrame
    df = pandas.DataFrame(logs_epoch)
    # 保存到 Excel 文件
    df.to_excel(excel_path, index=False)
    return log, df


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    # Save predictions and actual labels to a DataFrame  保存预测结果
    eval_df = pandas.DataFrame({
        'Original Label': actuals.flatten(),
        'Predicted Label': predictions.flatten()
    })
    eval_df.to_csv(evaluation_path, index=False)
    # Generate confusion matrix 画混淆矩阵
    # cm = confusion_matrix(actuals, predictions, labels=[1, 2, 3, 4, 5]) 错误，这里的label应该是编码后的标签
    cm = confusion_matrix(actuals, predictions, labels=[0, 1, 2, 3, 4])
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.xlabel('Predicted Label')
    plt.ylabel('Original Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(os.path.dirname(evaluation_path), 'confusion_matrix.png'))
    plt.show()

    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
def create_experiment_folder(base_path, exp_number):
    # Create a new folder for the experiment
    folder_path = os.path.join(base_path, f"exp{exp_number}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# prepare the data
# path = r"F:\results\variety1280\915newdata\merged_data.csv"
# 简化路径，自动生成文件路径存储
train_path = r"F:\results\variety1280\915newdata\data\train_addtime.csv"
test_path = r"F:\results\variety1280\915newdata\data\test_addtime.csv"
base_path = r"F:\results\variety1280\915newdata"
# Specify experiment number
# exp_number = 13
exp_number = 62
# Create experiment folder
exp_folder = create_experiment_folder(base_path, exp_number)
log_path = os.path.join(exp_folder, "training_log.csv")
evaluation_path = os.path.join(exp_folder, "evaluation_results.csv")
model_path = os.path.join(exp_folder, "model_weights.pth")
excel_path = os.path.join(exp_folder, 'training_epoch_logs.xlsx')

train_batch_size = 16
test_batch_size = 1024

train_dl, test_dl = prepare_data(train_path, test_path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(4)
print(model)
# train the model
epoch = 80
# 在主代码中捕获 log：确保 log 变量在主代码中被定义，并且你从 train_model 函数中获取了它。
log, df = train_model(train_dl, model, epoch)
train_model(train_dl, model, epoch)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
log['Accuracy'] = acc

# Save the training log to CSV
log_df = pandas.DataFrame({
    "Model Structure": [log["Model Structure"]],
    "Train Batch Size": [log["Train Batch Size"]],
    "Test Batch Size": [log["Test Batch Size"]],
    "Learning Rate": [log["Learning Rate"]],
    "Epochs": [log["Epochs"]],
    "Accuracy": [log["Accuracy"]]  # Losses need special handling
})
log_df.to_csv(log_path, index=False)

# save model
torch.save(model.state_dict(), model_path)

# make a single prediction
row = [5, 3, 1, 2]
# yhat = predict(row, model)
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# argmax(yhat) is index，not class
