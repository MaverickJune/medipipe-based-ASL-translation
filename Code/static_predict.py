import numpy as np
import torch
import torch.nn as nn

class static_model(nn.Module):
    def __init__(self):
        super(static_model, self).__init__()
        self.fc1 = nn.Linear(42, 32)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(32, 24)
        self.alphabet_index = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]        

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def test_print(self, x):
        with torch.no_grad():
            cord_numpy = np.array(x)
            cordinate_x = cord_numpy[:21]
            cordinate_y = cord_numpy[21:]
            cordinate_x = (cordinate_x - cordinate_x[0]) / (np.max(cordinate_x) - np.min(cordinate_x))
            cordinate_y = (cordinate_y - cordinate_y[0]) / (np.max(cordinate_y) - np.min(cordinate_y))
            cordinate = np.concatenate((cordinate_x, cordinate_y))
            cord = torch.from_numpy(cordinate).reshape((1, -1))
            out = self.forward(cord)
            _, predicted = torch.max(out, 1)
        return self.alphabet_index[predicted]
    def test_prob(self, x):
        with torch.no_grad():
            cord_numpy = np.array(x)
            cordinate_x = cord_numpy[:21]
            cordinate_y = cord_numpy[21:]
            cordinate_x = (cordinate_x - cordinate_x[0]) / (np.max(cordinate_x) - np.min(cordinate_x))
            cordinate_y = (cordinate_y - cordinate_y[0]) / (np.max(cordinate_y) - np.min(cordinate_y))
            cordinate = np.concatenate((cordinate_x, cordinate_y))
            cord = torch.from_numpy(cordinate).reshape((1, -1))
            out = self.forward(cord)
        return nn.functional.softmax(out)

            


