import torch

class HTRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # 🔥 flatten
        return self.fc(x)


class RetrievalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # 🔥 flatten
        return self.fc(x)