import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

food_type = ctrl.Antecedent(np.arange(0, 11, 1), 'food_type')
food_weight = ctrl.Antecedent(np.arange(0, 11, 1), 'food_weight')
temp = ctrl.Antecedent(np.arange(-10, 41, 1), 'temp')
water_content = ctrl.Antecedent(np.arange(0, 11, 1), 'water_content')
cooking_time = ctrl.Consequent(np.arange(0, 61, 1), 'cooking_time')
heat_level = ctrl.Consequent(np.arange(0, 101, 1), 'heat_level')

food_type.automf(3)
food_weight.automf(3)
temp.automf(3)
water_content.automf(3)
cooking_time.automf(3)
heat_level.automf(3)

rule1 = ctrl.Rule(food_weight['poor'] & food_type['poor'], cooking_time['poor'])
rule2 = ctrl.Rule(food_weight['average'] & food_type['average'], cooking_time['average'])
rule3 = ctrl.Rule(food_weight['good'] & food_type['good'], cooking_time['good'])
rule4 = ctrl.Rule(temp['poor'], heat_level['good'])
rule5 = ctrl.Rule(temp['average'], heat_level['average'])
rule6 = ctrl.Rule(temp['good'], heat_level['poor'])
rule7 = ctrl.Rule(water_content['poor'], cooking_time['good'])
rule8 = ctrl.Rule(water_content['average'], cooking_time['average'])
rule9 = ctrl.Rule(water_content['good'], cooking_time['poor'])

cooking_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
cooking_sim = ctrl.ControlSystemSimulation(cooking_ctrl)

cooking_sim.input['food_type'] = 5
cooking_sim.input['food_weight'] = 7
cooking_sim.input['temp'] = 25
cooking_sim.input['water_content'] = 6

cooking_sim.compute()

print(f"Cooking Time: {cooking_sim.output['cooking_time']:.2f} minutes")
print(f"Heat Level: {cooking_sim.output['heat_level']:.2f}%")
food_type.view()
food_weight.view()
temp.view()
water_content.view()
cooking_time.view()
heat_level.view()

