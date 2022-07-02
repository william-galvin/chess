# Author: William Galvin
# Since:  2022
# Version: 1.0

# This is a pytorch-based autoencoder for a chess position


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
from torch.nn import Sequential
from torch.nn import MSELoss
from torch.optim import SGD
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = "'cuda:0'" if torch.cuda.is_available() else "'cpu'"

EPOCHS = 15
FILE_NAME = "data.tsv"

class Autoencoder(Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encode_layers = Sequential(
            Linear(780, 600, device=device, dtype=torch.float32),
            ReLU(),
            Linear(600, 400, device=device, dtype=torch.float32),
            ReLU(),
            Linear(400, 200, device=device, dtype=torch.float32),
            ReLU(),
            Linear(200, 100, device=device, dtype=torch.float32)
        )

        self.decode_layers = Sequential(
            Linear(100, 200, device=device, dtype=torch.float32),
            ReLU(),
            Linear(200, 400, device=device, dtype=torch.float32),
            ReLU(),
            Linear(400, 600, device=device, dtype=torch.float32),
            ReLU(),
            Linear(600, 780, device=device, dtype=torch.float32)
        )

    def forward(self, x):
            x = self.encode_layers(x)
            x = self.decode_layers(x)
            return x

    def encode(self, x):
        return self.encode_layers(x)


class my_Dataset(Dataset):

    def load_table(self):
        return pandas.read_csv(FILE_NAME, sep = "\t")

    def __init__(self):
        print("loading tables...", end = '\r')
        self.table = self.load_table()
        print("loading complete", end = '\r')


    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        output = torch.tensor([self.table.iloc[idx, 2]], device=device, dtype=torch.float32)
        raw_input = self.table.iloc[idx, 1]
        input = eval("torch."+raw_input[:-1]+f", device={gpu})")
        return input, output    

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

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

losses = []
def train_model(model):
    model.train()
    start = time.time()
    optimizer = SGD(model.parameters(), lr = .002, momentum=.7, nesterov=True)
    criterion = MSELoss()
    train_dl, test_dl = prepare_data()
    for epoch in range(EPOCHS):
        print(f"{epoch}/{EPOCHS}\t time since start: {timeSince(start)}", end='\r')
        for i, (input, _) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(input)
            # calculate loss
            loss = criterion(yhat, input)
            losses.append(loss.cpu().detach().numpy())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
    print(f"\n\nmodel training complete in {timeSince(start)}")

def main():
    model = Autoencoder()
    train_model(model)

    model.eval()
    example = torch.rand(600, 780, device=device)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("autoencoder.pt")

    plt.plot(losses)
    plt.savefig("autoencoder-loss-plot")

if __name__ == "__main__":
    main()