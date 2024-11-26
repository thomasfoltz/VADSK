import json
import torch

from pishield.shield_layer import build_shield_layer

class ShieldLayer:
    def __init__(self, dataset, predictions):
        self.dataset = dataset
        self.num_variables = predictions.shape[-1]
        self.rules = {}
        self.rule_num = 0
        self.shield_layer = None
        self.predictions = predictions

    def construct_rule_statements(self):
        with open(f'{self.dataset}/rules.json', 'r') as file:
            rules_dict = json.load(file)

        for key, values in rules_dict.items():
            self.rules[key] = [f'y_{self.rule_num}']
            self.rule_num += 1
            self.rules[key].extend(f'y_{self.rule_num + i}' for i in range(len(values)))
            self.rule_num += len(values)

        with open(f'{self.dataset}/propositional_statements.txt', 'w') as file:
            for values in self.rules.values():
                file.write(' or '.join(values) + '\n')

    def init_shield_layer(self):
        ordering = ','.join(map(str, reversed(range(self.rule_num))))
        self.shield_layer = build_shield_layer(
            self.num_variables, 
            f'{self.dataset}/propositional_statements.txt', 
            ordering_choice='custom', 
            custom_ordering=ordering, 
            requirements_type='propositional'
        )

    def __call__(self):
        return self.shield_layer(self.predictions.clone())

predictions = torch.tensor([
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.]
])  # TAKEN FROM SHTECH DATASET EXAMPLE

shield_layer = ShieldLayer('SHTech', predictions)
shield_layer.construct_rule_statements()
shield_layer.init_shield_layer()
corrected_predictions = shield_layer()
print(corrected_predictions)

# TODO: look deeper into the shield layer stuff about the ordering
# TODO: see if I can abstract and simplify some of the code behind build_shield_layer to cater towards my needs
# TODO: understand how the logic actually works

