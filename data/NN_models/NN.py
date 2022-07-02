import torch
import time
import math
from torch.utils.data import Dataset
import pandas 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import BatchNorm1d
from torch.nn import Sequential
from torch.nn import CrossEntropyLoss
from torch.nn import Flatten
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np
from full_data import FenConverter
import random

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def load_table():
    return pandas.read_csv("fen_data.csv", sep = "|")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = "'cuda:0'" if torch.cuda.is_available() else "'cpu'"
EPOCHS = 100

class my_Dataset(Dataset):

    def __init__(self):
        print("loading tables...")
        self.table = load_table()
        self.converter = FenConverter()
        self.len = len(self.table)
        print("loading complete")


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        r1 = self.table.iloc[idx, 1]
        result1 = torch.tensor([int(r1)], device=device, dtype=torch.float32)
        fen1 = self.table.iloc[idx, 0]
        input1 = self.converter.fen_to_tensor(fen1)

        idx2 = random.randint(0, self.len - 1)
        r2 = self.table.iloc[idx2, 1]
        while r1 == r2:
            idx2 = random.randint(0, self.len - 1)
            r2 = self.table.iloc[idx2, 1]
        result2 = torch.tensor([int(r2)], device=device, dtype=torch.float32)
        fen2 = self.table.iloc[idx2, 0]
        input2 = self.converter.fen_to_tensor(fen2)
        
        return torch.cat((input1, input2)), torch.tensor([result1, result2], dtype=torch.float32, device=device)

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):

        # determine sizes
        test_size = round(n_test * self.__len__())
        train_size = self.__len__() - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

    def toTensor(self, tensor_string):
        return eval("torch."+tensor_string)


def prepare_data():
    data = my_Dataset()
    train, test = data.get_splits()
    train_dl = DataLoader(train, batch_size=256, shuffle=True)
    test_dl = DataLoader(test, batch_size=256, shuffle=False)
    return train_dl, test_dl

class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = Linear(780, 600, device=device, dtype=torch.float32)
        self.layer2 = Linear(600, 400, device=device, dtype=torch.float32)
        self.layer3 = Linear(400, 200, device=device, dtype=torch.float32)
        self.layer4 = Linear(200, 100, device=device, dtype=torch.float32)

        self.layers = Sequential(
            Linear(200, 100, device=device, dtype=torch.float32),
            BatchNorm1d(num_features=100, device=device, dtype=torch.float32),
            ReLU(),
            Linear(100, 64, device=device, dtype=torch.float32),
            BatchNorm1d(num_features=64, device=device, dtype=torch.float32),
            ReLU(),
            Linear(64, 16, device=device, dtype=torch.float32),
            BatchNorm1d(num_features=16, device=device, dtype=torch.float32),
            ReLU(),
            Linear(16, 2, device=device, dtype=torch.float32),
        )

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, input):
        x1, x2 = torch.chunk(input, 2, 1)
        x1 = self.encode(x1)
        x2 = self.encode(x2)

        return self.layers(torch.cat((x1, x2), dim=1))#.squeeze(-1)

losses = []
def train_model(model):
    start = time.time()
    optimizer = SGD(model.parameters(), lr = .002, momentum=.7, nesterov=True)
    criterion = CrossEntropyLoss()
    train_dl, test_dl = prepare_data()
    for epoch in range(EPOCHS):
        print(f"{epoch}/{EPOCHS}\t time since start: {timeSince(start)}")
        if epoch % 10 == 0:
            model.eval()
            example = torch.rand(600, 780*2, device=device) # where do these nums come from??
            traced_script_module = torch.jit.trace(model, example)
            traced_script_module.save("NN.pt")
            model.train()
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            losses.append(loss.cpu().detach().numpy())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
    print(f"\nmodel training complete in {timeSince(start)}")

def main():
    model = MLP()

    state_dict1 = torch.load('autoencoder_layer_1.pt')
    state_dict2 = torch.load('autoencoder_layer_2.pt')
    state_dict3 = torch.load('autoencoder_layer_3.pt')
    state_dict4 = torch.load('autoencoder_layer_4.pt')

    with torch.no_grad():
        model.layer1.weight.copy_(state_dict1['encode_layers.0.weight'])
        model.layer1.bias.copy_(state_dict1['encode_layers.0.bias'])

        model.layer2.weight.copy_(state_dict2['encode_layers.0.weight'])
        model.layer2.bias.copy_(state_dict2['encode_layers.0.bias'])

        model.layer3.weight.copy_(state_dict3['encode_layers.0.weight'])
        model.layer3.bias.copy_(state_dict3['encode_layers.0.bias'])

        model.layer4.weight.copy_(state_dict4['encode_layers.0.weight'])
        model.layer4.bias.copy_(state_dict4['encode_layers.0.bias'])

    train_model(model)
    model.eval()

    example = torch.rand(600, 780*2, device=device) # where do these nums come from??
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("NN.pt")

    plt.plot(losses)
    plt.savefig("loss-plot")

if __name__ == "__main__":
    main()
