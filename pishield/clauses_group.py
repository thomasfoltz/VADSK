import numpy as np

from pishield.literal import Literal
from pishield.clause import Clause
from pishield.constraints_group import ConstraintsGroup
from pishield.strong_coherency import strong_coherency_constraint_preprocessing

class ClausesGroup:
    def __init__(self, clauses):
        self.clauses = frozenset(clauses)
        self.clauses_list = clauses

    @classmethod
    def from_constraints_group(cls, group):
        return cls([Clause.from_constraint(cons) for cons in group])

    def __len__(self):
        return len(self.clauses)

    def __eq__(self, other):
        if not isinstance(other, ClausesGroup): return False
        return self.clauses == other.clauses

    def __add__(self, other):
        return ClausesGroup(self.clauses.union(other.clauses))

    def __str__(self):
        return '\n'.join([str(clause) for clause in self.clauses])

    def __hash__(self):
        return hash(self.clauses)

    def __iter__(self):
        return iter(self.clauses)

    def compacted(self):
        clauses = list(self.clauses)
        clauses.sort(reverse=True, key=len)
        compacted = []

        for clause in clauses:
            compacted = [c for c in compacted if not clause.is_subset(c)]
            compacted.append(clause)

        return ClausesGroup(compacted)

    def resolution(self, atom):
        pos = Literal(atom, True)
        neg = Literal(atom, False)

        # Split clauses in three categories
        pos_clauses, neg_clauses, other_clauses = set(), set(), set()
        for clause in self.clauses:
            if pos in clause:
                pos_clauses.add(clause)
            elif neg in clause:
                neg_clauses.add(clause)
            else:
                other_clauses.add(clause)

        # Apply resolution on positive and negative clauses
        resolution_clauses = [c1.resolution(c2, literal=pos) for c1 in pos_clauses for c2 in neg_clauses]
        resolution_clauses = {clause for clause in resolution_clauses if clause != None}
        next_clauses = ClausesGroup(other_clauses.union(resolution_clauses)).compacted()

        # Compute constraints 
        pos_constraints = [clause.fix_head(pos) for clause in pos_clauses]
        neg_constraints = [clause.fix_head(neg) for clause in neg_clauses]
        constraints = ConstraintsGroup(pos_constraints + neg_constraints)

        return constraints, next_clauses

    def stratify(self, centrality):
        # Centrality guides the inference order
        atoms = centrality

        # Apply resolution repeatedly
        group = ConstraintsGroup([])
        clauses = self

        for atom in atoms:
            # print(f"Eliminating %{atom} from %{len(clauses)} clauses\n")
            constraints, clauses = clauses.resolution(atom)
            if len(constraints.constraints_list):
                strongly_coherent_constraints = strong_coherency_constraint_preprocessing(constraints.constraints_list, atoms)
                if strongly_coherent_constraints is not None:
                    constraints = strongly_coherent_constraints
            group = group + constraints

        if len(clauses):
            raise Exception("Unsatisfiable set of clauses")

        return group.stratify()

    def coherent_with(self, preds):
        answer = [clause.coherent_with(preds) for clause in self.clauses_list]
        answer = np.array(answer).reshape(len(self.clauses_list), preds.shape[0])
        return answer.transpose()

    def atoms(self):
        result = set()
        for clause in self.clauses:
            result = result.union(clause.atoms())
        return result
