import torch.nn as nn
import pickle
from torch_geometric.nn.models import SchNet

class tmqm_SchNet(nn.Module):
    def __init__(self, hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=10.0,
                separated=False) -> None:
        super().__init__()

        self.separated=separated
        self.model = SchNet(
            hidden_channels,
            num_filters,
            num_interactions,
            num_gaussians,
            cutoff,
        )

    def forward(self,data):
        if self.separated:
            


if __name__ == '__main__':
    with open('../../dataset/data/complex.pkl','rb') as f:
        complexes = pickle.load(f)
    complexes