import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

load = ctrl.Antecedent(np.arange(0, 11, 1), 'load')
dirt = ctrl.Antecedent(np.arange(0, 11, 1), 'dirt')
fabric = ctrl.Antecedent(np.arange(0, 11, 1), 'fabric')
time = ctrl.Consequent(np.arange(0, 61, 1), 'time')
water = ctrl.Consequent(np.arange(0, 11, 1), 'water')

load['small'] = fuzz.trimf(load.universe, [0, 0, 5])
load['medium'] = fuzz.trimf(load.universe, [0, 5, 10])
load['large'] = fuzz.trimf(load.universe, [5, 10, 10])

dirt['low'] = fuzz.trimf(dirt.universe, [0, 0, 5])
dirt['medium'] = fuzz.trimf(dirt.universe, [0, 5, 10])
dirt['high'] = fuzz.trimf(dirt.universe, [5, 10, 10])

fabric['delicate'] = fuzz.trimf(fabric.universe, [0, 0, 5])
fabric['normal'] = fuzz.trimf(fabric.universe, [0, 5, 10])
fabric['heavy'] = fuzz.trimf(fabric.universe, [5, 10, 10])

time['short'] = fuzz.trimf(time.universe, [0, 0, 30])
time['medium'] = fuzz.trimf(time.universe, [0, 30, 60])
time['long'] = fuzz.trimf(time.universe, [30, 60, 60])

water['low'] = fuzz.trimf(water.universe, [0, 0, 5])
water['medium'] = fuzz.trimf(water.universe, [0, 5, 10])
water['high'] = fuzz.trimf(water.universe, [5, 10, 10])

rule1 = ctrl.Rule(load['small'] & dirt['low'] & fabric['delicate'], (time['short'], water['low']))
rule2 = ctrl.Rule(load['small'] & dirt['medium'] & fabric['normal'], (time['medium'], water['medium']))
rule3 = ctrl.Rule(load['medium'] & dirt['high'] & fabric['heavy'], (time['long'], water['high']))
rule4 = ctrl.Rule(load['large'] & dirt['high'] & fabric['heavy'], (time['long'], water['high']))
rule5 = ctrl.Rule(load['medium'] & dirt['medium'] & fabric['normal'], (time['medium'], water['medium']))

washing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
washing = ctrl.ControlSystemSimulation(washing_ctrl)

washing.input['load'] = 7
washing.input['dirt'] = 6
washing.input['fabric'] = 5

washing.compute()

print(f"Time: {washing.output['time']:.2f}")
print(f"Water level: {washing.output['water']:.2f}")
load.view()
dirt.view()
fabric.view()
time.view()
water.view()
