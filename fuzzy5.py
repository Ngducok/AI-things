import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

temp = ctrl.Antecedent(np.arange(16, 31, 1), 'temp')
humidity = ctrl.Antecedent(np.arange(10, 101, 10), 'humidity')
people = ctrl.Antecedent(np.arange(1, 6, 1), 'people')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
cooling = ctrl.Consequent(np.arange(0, 101, 1), 'cooling')

temp['cold'] = fuzz.trimf(temp.universe, [16, 16, 20])
temp['cool'] = fuzz.trimf(temp.universe, [18, 21, 24])
temp['comfortable'] = fuzz.trimf(temp.universe, [22, 24, 26])
temp['warm'] = fuzz.trimf(temp.universe, [24, 27, 30])
temp['hot'] = fuzz.trimf(temp.universe, [26, 30, 30])

humidity['low'] = fuzz.trimf(humidity.universe, [10, 10, 40])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

people['few'] = fuzz.trimf(people.universe, [1, 1, 3])
people['moderate'] = fuzz.trimf(people.universe, [2, 3, 4])
people['many'] = fuzz.trimf(people.universe, [3, 5, 5])

fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [30, 50, 70])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

cooling['low'] = fuzz.trimf(cooling.universe, [0, 0, 50])
cooling['medium'] = fuzz.trimf(cooling.universe, [30, 50, 70])
cooling['high'] = fuzz.trimf(cooling.universe, [50, 100, 100])

rule1 = ctrl.Rule(temp['cold'], [cooling['low'], fan_speed['low']])
rule2 = ctrl.Rule(temp['cool'], [cooling['low'], fan_speed['low']])
rule3 = ctrl.Rule(temp['comfortable'], [cooling['medium'], fan_speed['medium']])
rule4 = ctrl.Rule(temp['warm'], [cooling['high'], fan_speed['high']])
rule5 = ctrl.Rule(temp['hot'], [cooling['high'], fan_speed['high']])

rule6 = ctrl.Rule(humidity['low'], cooling['low'])
rule7 = ctrl.Rule(humidity['medium'], cooling['medium'])
rule8 = ctrl.Rule(humidity['high'], cooling['high'])

rule9 = ctrl.Rule(people['few'], fan_speed['low'])
rule10 = ctrl.Rule(people['moderate'], fan_speed['medium'])
rule11 = ctrl.Rule(people['many'], fan_speed['high'])

rule12 = ctrl.Rule(temp['hot'] & humidity['high'] & people['many'], [cooling['high'], fan_speed['high']])
rule13 = ctrl.Rule(temp['comfortable'] & humidity['medium'] & people['moderate'], [cooling['medium'], fan_speed['medium']])
rule14 = ctrl.Rule(temp['cool'] & humidity['low'] & people['few'], [cooling['low'], fan_speed['low']])

ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14])
ac = ctrl.ControlSystemSimulation(ac_ctrl)

ac.input['temp'] = 28
ac.input['humidity'] = 60
ac.input['people'] = 3

ac.compute()

print(f"Fan Speed: {ac.output['fan_speed']:.2f}")
print(f"Cooling Level: {ac.output['cooling']:.2f}")
temp.view()
humidity.view()
people.view()
fan_speed.view()
cooling.view()


