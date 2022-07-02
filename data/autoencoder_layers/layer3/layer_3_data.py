import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential
from torch.optim import SGD
import pandas

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class Autoencoder(Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encode_layers = Sequential(
            Linear(400, 200, device=device, dtype=torch.float32)
        )

        self.decode_layers = Sequential(
            Linear(200, 400, device=device, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.encode_layers(x)
        x = self.decode_layers(x)
        return x

    def encode(self, x):
        return self.encode_layers(x)

class Encoder():
    def __init__(self):
        self.model = Autoencoder()
        self.model.load_state_dict(torch.load("autoencoder_layer_3.pt"))
        self.model.to(device=device)
        self.model.eval()

        self.model.encode_layers.register_forward_hook(get_activation('encode_layers'))
        self.optimizer = SGD(self.model.parameters(), lr = .001, momentum=.7, nesterov=True)

    def extract_features(self, tensor):
        dataset = [tensor]
        input_dl = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, (inputs) in enumerate(input_dl):
                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                yhat = self.model(inputs)
        return activation['encode_layers']

def to_tensor(tensor_string):
    return eval("torch."+tensor_string)

def main():
    writer = open("layer_3_data.tsv", "w")
    encoder = Encoder()
    FILE_NAME = "layer_2_data.tsv"
    with open(FILE_NAME) as f:
        for line_ in f:
            line = line_.split("\t")
            t = to_tensor(line[0])
            writer.write(str(encoder.extract_features(t)).replace("\n", "").replace("         ", " ") + '\n')




if __name__ == "__main__":
    main()
