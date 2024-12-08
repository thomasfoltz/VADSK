import numpy as np

from pishield.literal import Literal
from pishield.constraint import Constraint

class Clause:
    def __init__(self, literals):
        if isinstance(literals, str):
            # Clause(string)
            literals = [Literal(lit) for lit in literals.split(' ')]
            self.literals = frozenset(literals)
        else:
            # Clause([Literals])
            self.literals = frozenset(literals)

    def __len__(self):
        return len(self.literals)

    def __iter__(self):
        return iter(self.literals)

    def __eq__(self, other):
        if not isinstance(other, Clause): return False
        return self.literals == other.literals

    def __hash__(self):
        return hash(self.literals)

    def __str__(self):
        return ' '.join([str(literal) for literal in sorted(self.literals)])

    @classmethod
    def from_constraint(cls, constraint):
        body = [lit.neg() for lit in constraint.body]
        return cls([constraint.head] + body)

    def fix_head(self, head):
        if not head in self.literals:
            raise Exception('Head not in clause')
        body = [lit.neg() for lit in self.literals if lit != head]
        return Constraint(head, body)

    def always_true(self):
        for literal in self.literals:
            if literal.neg() in self.literals:
                return True
        return False

    def resolution_on(self, other, literal):
        result = self.literals.union(other.literals).difference({literal, literal.neg()})
        result = Clause(result)
        return None if result.always_true() else result

    def resolution(self, other, literal=None):
        if literal != None:
            return self.resolution_on(other, literal)

        for lit in self.literals:
            if lit.neg() in other.literals:
                return self.resolution_on(other, lit)

        return None

    def coherent_with(self, preds):
        pos = [lit.atom for lit in self.literals if lit.positive]
        neg = [lit.atom for lit in self.literals if not lit.positive]

        preds = np.concatenate((preds[:, pos], 1 - preds[:, neg]), axis=1)
        preds = preds.max(axis=1)
        return preds > 0.5

    def is_subset(self, other):
        return self.literals.issubset(other.literals)

    def atoms(self):
        return {lit.atom for lit in self.literals}
