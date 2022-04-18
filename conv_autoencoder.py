# https://machinelearningknowledge.ai/pytorch-conv2d-explained-with-examples/#iv_Exploring_Dataset

import math
import time
import torch
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import Sequential
from torch.nn import Flatten
from torch.nn import ReLU
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ConvTranspose2d
from torch.nn import ConvTranspose3d
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import chess
import pandas
import matplotlib.pyplot as plt

FILE_NAME = "small_fen_data.csv"
EPOCHS = 1
BATCH_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def position_to_conv_tensor(fen):
    """
    Converts a chess position to a tensor.
    """
    position = chess.BaseBoard()
    position.set_board_fen(fen[:fen.index(' ')])
    tensor = torch.zeros(2, 6, 8, 8, device=device, dtype=torch.float32)
    for i in range(8):
        for j in range(8):
            piece = position.piece_at(chess.square(i, j))
            if piece is not None:
                if piece.color == chess.WHITE:
                    tensor[0, piece.piece_type-1, i, j] = 1
                else:
                    tensor[1, piece.piece_type-1, i, j] = 1
    return tensor

class Autoencoder(Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Chess board can be represted as tensors of dimensions 8*8*6*2, where there are 2 channels (one for white, one for black), 6 pieces types represented as 1 or 0, and an 8*8 board
        self.encode_layers = Sequential(
            # convolutional layers
            Conv3d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Flatten()
        )
        self.encode_layers_2 = Sequential(
            # Fully cinnected linear layers that result in a flat tensor on length 100
            Flatten(),
            Linear(in_features=32*8*8*6, out_features=4096, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Linear(in_features=4096, out_features=1024, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Linear(in_features=1024, out_features=512, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Linear(in_features=512, out_features=100, device=device, dtype=torch.float32)
        )
        
        self.decode_layers = Sequential(
            # Fully cinnected linear layers that result in a flat tensor on length 100 
            Linear(in_features=100, out_features=512, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Linear(in_features=512, out_features=1024, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            Linear(in_features=1024, out_features=4096, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            # convolutional layers
            Linear(in_features=4096, out_features=32*8*8*6, device=device, dtype=torch.float32),
            ReLU(inplace=True)
        )
        self.decode_layers_2 = Sequential(
            ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.float32),
            ReLU(inplace=True),
            ConvTranspose3d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.float32),
            ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.encode_layers(x)
        x = self.encode_layers_2(x)
        x = torch.flatten(x)
        x = self.decode_layers(x)
        size = int(x.size()[0]*x[0].size()[0]/(32*6*8*8))
        x = torch.reshape(x, [size, 32, 6, 8, 8])
        x = self.decode_layers_2(x)
        return x


class my_Dataset(Dataset):

    def load_table(self):
        return pandas.read_csv(FILE_NAME, sep = "|")

    def __init__(self):
        print("loading tables...", end = '\r')
        self.table = self.load_table()
        print("loading complete", end = '\r')


    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        output = torch.tensor([self.table.iloc[idx, 1]], device=device, dtype=torch.float32)
        fen = self.table.iloc[idx, 0]
        input = position_to_conv_tensor(fen)
        return input, output    

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * self.__len__())
        train_size = self.__len__() - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

def prepare_data():
    data = my_Dataset()
    train, test = data.get_splits()
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    return train_dl, test_dl


losses = []
def train_model(model):
    model.train()
    start = time.time()
    criterion = torch.nn.MSELoss()    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=.002)
    
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
    example = torch.rand(2, 6, 8, 8)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("autoencoder.pt")

    plt.plot(losses)
    plt.savefig("autoencoder-loss-plot")

if __name__ == "__main__":
    main()


def main():
    
    start = time.time()
    table = pandas.read_csv(FILE_NAME, sep = "|")
    for _ in range(100):
        for _, row in table.iterrows():
            position_to_conv_tensor(row[0])
    print(f"conv: {timeSince(start)}")

main()

    
    

#! add "HOOKS"