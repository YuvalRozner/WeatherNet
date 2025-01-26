# model.py

import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, label_column,label_width):
        super(Baseline, self).__init__()
        self.label_column = label_column
        self.label_width = label_width


    def forward(self, x):
        # x shape is (batch, time, features)
        tempPrevDay = x[:, -1, self.label_column]  
        #if self.label_width > 1:
         #   return tempPrevDay.repeat(self.label_width)
        return tempPrevDay