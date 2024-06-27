import torch.nn as nn


class SequentialNN(nn.Module):
    def __init__(self, feature_count, num_classes):
        super(SequentialNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_count, feature_count),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_count, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
