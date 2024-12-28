## features include but are not limited to,
# percentage of the question given, the
# previous occurrences of this guess,
# and finally confidence


import torch.nn 
import numpy as np


class Buzzer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = [[1,0,0]]
        
        l_1 = torch.nn.Linear(3,1)
        l_1.weight = torch.nn.Parameter(torch.tensor(self.w, dtype=torch.float32))
        l_1.bias = torch.nn.Parameter(torch.tensor([0.0], dtype= torch.float32))
        self.nn = torch.nn.Sequential(
            l_1
        )

    def forward(self, x):
        return self.nn(x)


    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    model = Buzzer() 
    input_tensor = torch.randn(15, 3) 
    output_tensor = model(input_tensor)
