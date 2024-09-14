## features include but are not limited to, 
# persentage of the question given, the 
# previous occurances of this guess, 
# and finally confidence
import torch
import numpy as np


class Buzzer(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(np.random.rand(1, 4))
        lin0 = torch.nn.Linear(4, 1)
        act0 = torch.nn.ReLU()
        lin1 = torch.nn.Linear(4, 1)
        act1 = torch.nn.ReLU() 
        self.model = torch.nn.Sequential(lin0, act0, lin1, act1)

        self.guesses = {}
    def forward(self):
        pass

    def __call__(self, conf: int, num_occ: int, time: float) -> list[int]:
        

        return self.model()
    def train(self):
        pass
    def save(self):
        pass
    def load(self):
        pass

