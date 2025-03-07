import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

traffic = ctrl.Antecedent(np.arange(0, 11, 1), 'traffic')
distance = ctrl.Antecedent(np.arange(0, 11, 1), 'distance')
order_time = ctrl.Antecedent(np.arange(0, 11, 1), 'order_time')
performance = ctrl.Antecedent(np.arange(0, 11, 1), 'performance')
delivery_time = ctrl.Consequent(np.arange(0, 11, 1), 'delivery_time')
driver_rating = ctrl.Consequent(np.arange(0, 11, 1), 'driver_rating')

traffic.automf(3)
distance.automf(3)
order_time.automf(3)
performance.automf(3)
delivery_time.automf(3)
driver_rating.automf(3)

rule1 = ctrl.Rule(traffic['poor'] & distance['poor'], delivery_time['poor'])
rule2 = ctrl.Rule(traffic['average'] & distance['average'], delivery_time['average'])
rule3 = ctrl.Rule(traffic['good'] & distance['good'], delivery_time['good'])
rule4 = ctrl.Rule(order_time['poor'], driver_rating['poor'])
rule5 = ctrl.Rule(order_time['average'], driver_rating['average'])
rule6 = ctrl.Rule(order_time['good'], driver_rating['good'])
rule7 = ctrl.Rule(performance['poor'], driver_rating['poor'])
rule8 = ctrl.Rule(performance['average'], driver_rating['average'])
rule9 = ctrl.Rule(performance['good'], driver_rating['good'])

rating_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
rating_sim = ctrl.ControlSystemSimulation(rating_ctrl)

rating_sim.input['traffic'] = 6
rating_sim.input['distance'] = 4
rating_sim.input['order_time'] = 7
rating_sim.input['performance'] = 8

rating_sim.compute()

print(f"Delivery Time: {rating_sim.output['delivery_time']:.2f} minutes")
print(f"Driver Rating: {rating_sim.output['driver_rating']:.2f}/10")
traffic.view()
distance.view()
order_time.view()
performance.view()
delivery_time.view()
driver_rating.view()