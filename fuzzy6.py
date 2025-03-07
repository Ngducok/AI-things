import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

input_1 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_1')
input_2 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_2')
input_3 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_3')
output = ctrl.Consequent(np.arange(0, 11, 1), 'output')

input_1['low'] = fuzz.trimf(input_1.universe, [0, 0, 5])
input_1['medium'] = fuzz.trimf(input_1.universe, [0, 5, 10])
input_1['high'] = fuzz.trimf(input_1.universe, [5, 10, 10])

input_2['low'] = fuzz.trimf(input_2.universe, [0, 0, 5])
input_2['medium'] = fuzz.trimf(input_2.universe, [0, 5, 10])
input_2['high'] = fuzz.trimf(input_2.universe, [5, 10, 10])

input_3['low'] = fuzz.trimf(input_3.universe, [0, 0, 5])
input_3['medium'] = fuzz.trimf(input_3.universe, [0, 5, 10])
input_3['high'] = fuzz.trimf(input_3.universe, [5, 10, 10])

output['low'] = fuzz.trimf(output.universe, [0, 0, 5])
output['medium'] = fuzz.trimf(output.universe, [0, 5, 10])
output['high'] = fuzz.trimf(output.universe, [5, 10, 10])

rule1 = ctrl.Rule(input_1['low'] & input_2['low'] & input_3['low'], output['low'])
rule2 = ctrl.Rule(input_1['medium'] & input_2['medium'] & input_3['medium'], output['medium'])
rule3 = ctrl.Rule(input_1['high'] & input_2['high'] & input_3['high'], output['high'])

system_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
system = ctrl.ControlSystemSimulation(system_ctrl)

system.input['input_1'] = 7
system.input['input_2'] = 5
system.input['input_3'] = 3

system.compute()

print(system.output['output'])
input_1.view()
input_2.view()
input_3.view()
output.view()



