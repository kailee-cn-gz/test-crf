import torch.nn as nn
from config import *
from torchcrf import CRF
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, TARGET_SIZE)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)

    def get_lstm_param(self, input):
        out = self.embedding(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self.get_lstm_param(input)
        return out


if __name__ == '__main__':
    model = Model()
    #测试
    input=torch.randint(1, 3000, (100, 50))
    print(model)
    print(model(input))
