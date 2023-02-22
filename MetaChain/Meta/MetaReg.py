import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import os
from sklearn import metrics
from sklearn.model_selection import KFold
import argparse
import glob

# set random seed
seed = 42
torch.manual_seed(seed)

import learn2learn as l2l


class BlockChainDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data).type(torch.float32)
        self.y_data = torch.from_numpy(label).type(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class MetaBlockChainDataset(Dataset):
    def __init__(self, data, label, trainloader=None):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data).type(torch.float32)
        self.y_data = torch.from_numpy(label).type(torch.float32)
        self.loader = trainloader

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# design model using class
class SineModel(torch.nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 8)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def targetdata(targetdataset, task):
    df = pd.read_csv('../data/BPD' + str(targetdataset) + '.csv')
    xy = df.valuesxy = df.values
    raw_data = xy
    X = raw_data[:, :2]
    if task == 1:
        Y1 = raw_data[:, 2].reshape((-1, 1))
    else:
        Y1 = raw_data[:, 3].reshape((-1, 1))

    data = np.hstack((X, Y1))
    # np.random.shuffle(data)
    X = data[:, :2]
    Y1 = data[:, 2].reshape((-1, 1))
    return X, Y1


def computacc(maml, Xtrain1, Xtest1, Ytrain1, Ytest1,test_adapt_steps):
    print('Iteration:', iter, 'Meta Train Loss', meta_train_loss.item())

    Xtrain1_minmax = min_max_scaler1.fit_transform(Xtrain1)
    # Apply the same scaling to the test set data
    Xtest1_minmax = min_max_scaler1.transform(Xtest1)

    Xtest1_tensor = torch.from_numpy(Xtest1_minmax).type(torch.float32)
    Ytest1_tensor = torch.from_numpy(Ytest1).type(torch.float32)

    train_dataset = BlockChainDataset(Xtrain1_minmax, Ytrain1)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    model = SineModel()
    model.load_state_dict(maml.clone().module.state_dict())
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(test_adapt_steps):
        for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # support_preds = learner(x_support_tensor)
        # support_loss = lossfn(support_preds, y_support_tensor)
        # learner.adapt(support_loss)

    y_pred = model(Xtest1_tensor).data.numpy()
    y = Ytest1_tensor.numpy()
    MAE = metrics.mean_absolute_error(y, y_pred)
    RMSE = metrics.mean_squared_error(y, y_pred) ** 0.5
    MAPE = metrics.mean_absolute_percentage_error(y, y_pred)
    return MAE, RMSE, MAPE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MetaChain')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--numbermodel', type=int, default=3)
    parser.add_argument('--adapt_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--adapt_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=5)
    parser.add_argument('--test_adapt_steps', type=int, default=10000)
    parser.add_argument('--targetdataset', type=int, default=5, help='Indicates which data set to use')
    parser.add_argument('--task', type=int, default=1, help='Indicates which task to perform')
    args = parser.parse_args()

    epochs = args.epochs
    numbermodel = args.numbermodel
    adapt_lr = args.adapt_lr
    meta_lr = args.meta_lr
    adapt_steps = args.adapt_steps
    batch_size = args.batch_size
    resultdir = 'result'
    results = []
    bestMSE1 = [[500, 500, 500], [500, 500, 500], [500, 500, 500], [500, 500, 500], [500, 500, 500], ]

    testname = [1, 2, 3, 4, 5]
    testname.remove(args.targetdataset)

    #每次都将除去自己的其它数据集用于元训练
    X1, Y1 = targetdata(testname[0], args.task)
    X2, Y2 = targetdata(testname[1], args.task)
    X3, Y3 = targetdata(testname[2], args.task)
    Xtrain1 = np.vstack((X1, np.vstack((X2, X3))))
    Ytrain1 = np.vstack((Y1.reshape((-1, 1)), np.vstack((Y2.reshape((-1, 1)), Y3.reshape((-1, 1))))))

    # 数据归一化
    min_max_scaler1 = preprocessing.MinMaxScaler()
    Xtrain1_minmax = min_max_scaler1.fit_transform(Xtrain1)

    # 获取训练数据
    Xtest1_minmax, Ytest1 = targetdata(args.targetdataset, args.task)


    model = SineModel()
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')
    data = np.hstack((Xtrain1_minmax, Ytrain1))
    for epoch in trange(epochs):
        np.random.shuffle(data)
        Xtrain1_minmax = data[:, :2]
        Ytrain1 = data[:, 2].reshape((-1, 1))
        # 对数据进行划分
        metaX = []
        metaY = []
        midx = []
        midy = []
        for i in range(Xtrain1_minmax.shape[0]):
            midx.append(Xtrain1_minmax[i])
            midy.append(Ytrain1[i])
            if len(midx) == batch_size:
                metaX.append(np.array(midx))
                metaY.append(np.array(midy))
                midx = []
                midy = []
        metaX = np.array(metaX)
        metaY = np.array(metaY)
        traindata = BlockChainDataset(metaX, metaY)
        train_loader = DataLoader(dataset=traindata, batch_size=numbermodel, shuffle=True, pin_memory=True,
                                  drop_last=True)
        # for each iteration
        for iter, batch in enumerate(train_loader):  # num_tasks/batch_size
            meta_train_loss = 0.0

            # for each task in the batch
            effective_batch_size = batch[0].shape[0]
            for i in range(effective_batch_size):
                learner = maml.clone()

                # divide the data into support and query sets
                train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
                x_support, y_support = train_inputs[::2], train_targets[::2]
                x_query, y_query = train_inputs[1::2], train_targets[1::2]

                for _ in range(adapt_steps):  # adaptation_steps
                    support_preds = learner(x_support)
                    support_loss = lossfn(support_preds, y_support)
                    learner.adapt(support_loss)

                query_preds = learner(x_query)
                query_loss = lossfn(query_preds, y_query)
                meta_train_loss += query_loss

            meta_train_loss = meta_train_loss / effective_batch_size

            opt.zero_grad()
            meta_train_loss.backward()
            opt.step()
        #使用外部数据集训练的元模型用于初始化参数
        if epoch % 100 == 0:
            KF = KFold(n_splits=5, random_state=seed, shuffle=True)
            i=0
            for train_index, test_index in KF.split(Xtest1_minmax):
                Xtrain2, Xtest2 = Xtest1_minmax[test_index[:args.test_batch_size]], Xtest1_minmax[
                    np.vstack((train_index.reshape((-1, 1)), test_index[args.test_batch_size:].reshape((-1, 1)))).reshape((-1))]
                Ytrain2, Ytest2 = Ytest1[test_index[:args.test_batch_size]], Ytest1[
                    np.vstack((train_index.reshape((-1, 1)), test_index[args.test_batch_size:].reshape((-1, 1)))).reshape((-1))]

                # 在此处初始化超参数并对目标数据集进行测试
                MAE1, RMSE1, MAPE1 = computacc(maml, Xtrain2, Xtest2, Ytrain2, Ytest2,
                                               args.test_adapt_steps)
                print(f'dataset1---   MAE:{MAE1} RMSE:{RMSE1} MAPE:{MAPE1} '.format(MAE=MAE1, RMSE=RMSE1, MAPE=MAPE1))
                if MAE1 < bestMSE1[i][0]:
                    bestMSE1[i] = [MAE1, RMSE1, MAPE1]
                i=i+1
    bestMSE1 = np.array(bestMSE1)
    #保存结果
    resultname = str(args.numbermodel) + '_' + str(args.adapt_steps) + '_' + str(args.batch_size) + '_' + str(
        args.meta_lr) + '_' + str(args.adapt_lr) + '_' + str(args.test_batch_size) + '_' + str(
        args.test_adapt_steps) + '.csv'
    os.makedirs(os.path.join(str(args.targetdataset), str(args.task)), exist_ok=True)
    pd.DataFrame(np.vstack((np.vstack((bestMSE1, bestMSE1.mean(0).reshape((1, -1)))), bestMSE1.std(0)))).to_csv(
        os.path.join(str(args.targetdataset), str(args.task), resultname))
