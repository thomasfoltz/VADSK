class Literal:
    def __init__(self, *args, reversed_sign=False):
        if len(args) == 2:
            # Literal(int, bool)
            self.atom = args[0]
            self.positive = args[1]
        else:
            # Literal(string)
            plain = args[0]
            if 'y_' in plain:
                if 'not ' in plain:
                    self.atom = int(plain[6:])
                    if reversed_sign:
                        self.positive = True   # set to True, to account for the :- format, which the code uses
                    else:
                        self.positive = False   # set to True, to account for the :- format, which the code uses
                else:
                    self.atom = int(plain[2:])
                    if reversed_sign:
                        self.positive = False  # set to False, to account for the :- format, which the code uses
                    else:
                        self.positive = True
            else:
                if 'n' in plain:
                    self.atom = int(plain[1:])
                    self.positive = False
                else:
                    self.atom = int(plain)
                    self.positive = True

    def __str__(self):
        return str(self.atom) if self.positive else 'n' + str(self.atom)

    def __hash__(self):
        return hash((self.atom, self.positive))

    def __eq__(self, other):
        if not isinstance(other, Literal): return False
        return self.atom == other.atom and self.positive == other.positive

    def __lt__(self, other):
        if self.atom == other.atom:
            return self.positive < other.positive
        else:
            return self.atom < other.atom

    def neg(self):
        return Literal(self.atom, not self.positive)
