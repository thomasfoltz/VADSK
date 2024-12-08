import torch
from torch import nn
import numpy as np

from pishield.clauses_group import ClausesGroup
from pishield.constraints_group import ConstraintsGroup
from pishield.constraints_module import ConstraintsModule

class ShieldLayer(nn.Module):
    def __init__(self, num_classes: int,
                 requirements: str = None,
                 ordering: str = list,):
        super(ShieldLayer, self).__init__()

        self.num_classes = num_classes
        self.ordering = ordering
        constraints_filepath = requirements

        constraints_group = ConstraintsGroup(constraints_filepath)
        clauses_group = ClausesGroup.from_constraints_group(constraints_group)

        centrality = np.array(self.ordering)
        strata = clauses_group.stratify(centrality)
        self.stratified_constraints = strata

        # print(f"Generated {len(strata)} strata of constraints with {centrality} centrality")

        self.atoms = nn.Parameter(torch.tensor(list(range(num_classes))), requires_grad=False)

        modules = [ConstraintsModule(stratum, num_classes) for stratum in strata]
        self.module_list = nn.ModuleList(modules)

        core = set(range(num_classes))
        strata = [stratum.heads() for stratum in strata]
        for stratum in strata:
            core = core.difference(stratum)
        assert len(core) > 0

    def to_minimal(self, tensor):
        return tensor[:, self.atoms].reshape(tensor.shape[0], len(self.atoms))

    def from_minimal(self, tensor, init):
        return init.index_copy(1, self.atoms, tensor)

    def forward(self, preds):
        updated = self.to_minimal(preds)

        for module in self.module_list:
            updated = module(updated, goal=None, iterative=True)

        return self.from_minimal(updated, preds)
