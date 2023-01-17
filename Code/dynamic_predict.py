import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class Dynamic_predict(nn.Module):
    def __init__(self):
        super(Dynamic_predict, self).__init__()
        self.lstm = nn.LSTM(input_size = 42, hidden_size = 10, batch_first = True, num_layers = 1)
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.h = None
        self.c = None

    def forward(self, x):
        h0 = torch.zeros(1, 1, 10)
        c0 = torch.zeros(1, 1, 10)
        out, (h, c)= self.lstm(x, (h0, c0))
        #print(out[-1].shape)
        out = self.fc(out[-1][-1])
        return self.sigmoid(out)

    def test_print(self, x):
        x = np.array(x)
        hand_x = x[:, :21]
        hand_y = x[:, 21:]
        hand_x = (hand_x - np.min(hand_x)) / (np.max(hand_x) - np.min(hand_x))
        hand_y = (hand_y - np.min(hand_y)) / (np.max(hand_y) - np.min(hand_y))
        cordinate = np.concatenate((hand_x, hand_y))
        cordinate = np.reshape(cordinate, (1, -1, 42))
        cordinate = torch.from_numpy(cordinate)
        with torch.no_grad():
            out = self.forward(cordinate)
            if out < 0.5:
                return "J"
            else :
                return "Z"

