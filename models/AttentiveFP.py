import sys
import torch
sys.path.append('./models')
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import global_add_pool

num_atom_type = 119

class tmqm_AttentiveFP(AttentiveFP):
    def forward(self, x, edge_index, edge_attr,batch,separated=False,attention=False) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        if attention:
            return x
        else:
            row = torch.arange(batch.size(0), device=batch.device)
            edge_index = torch.stack([row, batch], dim=0)

            out = global_add_pool(x, batch).relu_()
            for t in range(self.num_timesteps):
                h = F.elu_(self.mol_conv((x, out), edge_index))
                h = F.dropout(h, p=self.dropout, training=self.training)
                out = self.mol_gru(h, out).relu_()
            if separated == False:
                # Predictor:
                out = F.dropout(out, p=self.dropout, training=self.training)
                return self.lin2(out)
            else:
                return out

        