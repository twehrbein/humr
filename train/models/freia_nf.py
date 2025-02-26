from FrEIA.framework import (
    InputNode,
    OutputNode,
    Node,
    ReversibleGraphNet,
    ConditionNode,
)
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import torch.nn as nn


def cond_realnvp_nf(input_dim, n_blocks, clamp, fc_size, cond_size, dropout=0.0):

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, fc_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(fc_size, fc_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(fc_size, dims_out),
        )

    nodes = [InputNode(input_dim, name="input")]
    cond = ConditionNode(cond_size, name="condition")
    for k in range(n_blocks):
        nodes.append(
            Node(
                nodes[-1],
                GLOWCouplingBlock,
                {"subnet_constructor": subnet_fc, "clamp": clamp},
                conditions=cond,
                name=f"coupling_{k}",
            )
        )
        nodes.append(Node(nodes[-1], PermuteRandom, {"seed": k}, name=f"permute_{k}"))

    nodes.append(OutputNode(nodes[-1], name="output"))
    return ReversibleGraphNet(nodes + [cond], verbose=False)
