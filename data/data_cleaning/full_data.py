import chess
import torch

from layer_1_data import Encoder as Layer1
from layer_2_data import Encoder as Layer2
from layer_3_data import Encoder as Layer3
from layer_4_data import Encoder as Layer4

class Encoder:
    def __init__(self):
        self.layers = [Layer1(), Layer2(), Layer3(), Layer4()]
        self.converter = FenConverter()

    def extract_features(self, tensor):
        for layer in self.layers:
            tensor = layer.extract_features(tensor)[0]
        return tensor

    def fen_to_features(self, fen):
        tensor = self.converter.fen_to_tensor(fen)
        return self.extract_features(tensor)


class FenConverter:
    def __init__(self):
        self.all_letters = ['p', 'P', 'n', 'N', 'b', 'B', 'r', 'R', 'q', 'Q', 'k', 'K']
        self.n_letters = len(self.all_letters)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bboard = chess.BaseBoard()

    def letterToIndex(self, letter):
        return self.all_letters.index(letter)

    def charToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters, dtype=torch.float32, device=self.device)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return (tensor[0][0])

    def fen_to_tensor(self, FEN: str):
        self.bboard.set_board_fen(FEN[:FEN.index(' ')])
        turn = FEN[FEN.index(' ')+1]

        dict = self.bboard.piece_map()
        list = []
        for i in range(0, 64):
            if i in dict:
                list.append(self.charToTensor(dict[i].symbol()))
            else:
                list.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32, device=self.device))

        if turn == 'w':
            list.append(torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32, device=self.device))
        else:
            list.append(torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32, device=self.device))

        return torch.flatten(torch.stack(list))

def main():
    encoder = FenConverter()
    print(encoder.fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))

if __name__ == "__main__":
    main()
