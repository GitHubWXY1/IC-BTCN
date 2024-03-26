import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.functional import confusion_matrix


from datasetloader.dataloader_3 import Dataset_Mooc


from Model.model import IC_BTCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256
Path = "../dataset/npy_3_CLIR/"

net = IC_BTCN()
# net = CNN_LSTMNet()
# net = TCNet()
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
save_path = "./pth/CBiTCN_3_CLIR_lr001_.pth"
# save_path = "./CNN_LSTM_3_CLIR_lr0005.pth"
# save_path = "./Lenet_3_CLIR_lr001.pth"
dataset_train = Dataset_Mooc(Path,  flag="train")
train_data = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = Dataset_Mooc(Path,  flag="test")
test_data = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
def Ttrain():
    print("train:-----------------------------------------------------------------")
    for epoch in range(100):
        running_loss = 0.0
        running_Evaluate = np.zeros(4)
        len = 0
        for step, data in enumerate(train_data, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            outputs.to(device)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            predict_y = torch.max(outputs, dim=1)[1]

            eva = evall(predict_y.to("cpu"), labels.to("cpu"))
            running_Evaluate +=eva
            len+=1
        eva = running_Evaluate / len
        print("train acc, precision, reall, F1 :", eva)
        print("epoch：", epoch, "loss is", running_loss)
    torch.save(net.state_dict(), save_path)
    print("finished Training , loss is", running_loss)

def ttest():
    net.load_state_dict(torch.load(save_path))
    print("test:-----------------------------------------------------------------")
    with torch.no_grad():
        running_Evaluate = np.zeros(4)
        len = 0
        for step, data in enumerate(train_data, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            predict_y = torch.max(outputs, dim=1)[1]

            eva = evall(predict_y.to("cpu"), labels.to("cpu"))
            # acc = (predict_y == labels).sum().item() / labels.size(0)
            running_Evaluate +=eva
            len+=1

        eva = running_Evaluate/len
        print("test acc, precision, reall, F1 :", eva)

def swap(arr):
    temp = arr[0][0]
    arr[0][0] = arr[1][1]
    arr[1][1] = temp

    temp = arr[0][1]
    arr[0][1] = arr[1][0]
    arr[1][0] = temp
    return arr

def evall(pres, labels):
    conf_matrix = np.zeros((2, 2))
    p = pres.numpy()
    l = labels.numpy()
    for pp, tt in zip(p, l):
        conf_matrix[tt, pp] +=1
    conf_matrix = swap(conf_matrix)
    # 0-0
    TP = conf_matrix[0][0]
    # 0-1
    FP = conf_matrix[0][1]
    # 1-0
    FN = conf_matrix[1][0]
    # 1-1
    TN = conf_matrix[1][1]

    acc = (TP +TN) /  (TP+TN +FP + FN)
    precision = TP / (TP + FP)
    reall = TP / (TP + FN)
    F1 = 2*TP /(2*TP + FP + FN)
    return np.array([acc, precision, reall, F1])


# Ttrain()

import time
start_time = time.time()

ttest()

end_time = time.time()
elapsed_time_ms = (end_time - start_time) * 1000
print(f"time {elapsed_time_ms:.2f} ms")