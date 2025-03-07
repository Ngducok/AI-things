import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

market_demand = ctrl.Antecedent(np.arange(0, 11, 1), 'market_demand')
competitor_price = ctrl.Antecedent(np.arange(0, 11, 1), 'competitor_price')
production_cost = ctrl.Antecedent(np.arange(0, 11, 1), 'production_cost')
seasonal_factor = ctrl.Antecedent(np.arange(0, 11, 1), 'seasonal_factor')
stock_level = ctrl.Antecedent(np.arange(0, 11, 1), 'stock_level')
pricing_decision = ctrl.Consequent(np.arange(0, 11, 1), 'pricing_decision')

market_demand.automf(3)
competitor_price.automf(3)
production_cost.automf(3)
seasonal_factor.automf(3)
stock_level.automf(3)
pricing_decision.automf(3)

rule1 = ctrl.Rule(market_demand['poor'] & competitor_price['poor'] & stock_level['good'], pricing_decision['poor'])
rule2 = ctrl.Rule(market_demand['average'] & production_cost['average'] & stock_level['good'], pricing_decision['average'])
rule3 = ctrl.Rule(market_demand['good'] & seasonal_factor['good'] & stock_level['poor'], pricing_decision['good'])
rule4 = ctrl.Rule(market_demand['poor'] & competitor_price['average'] & stock_level['average'], pricing_decision['poor'])
rule5 = ctrl.Rule(market_demand['good'] & production_cost['poor'] & stock_level['average'], pricing_decision['good'])
rule6 = ctrl.Rule(market_demand['average'] & competitor_price['good'] & seasonal_factor['average'], pricing_decision['average'])
rule7 = ctrl.Rule(market_demand['good'] & competitor_price['good'] & stock_level['poor'], pricing_decision['good'])

pricing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
pricing_sim = ctrl.ControlSystemSimulation(pricing_ctrl)

pricing_sim.input['market_demand'] = 6
pricing_sim.input['competitor_price'] = 4
pricing_sim.input['production_cost'] = 5
pricing_sim.input['seasonal_factor'] = 7
pricing_sim.input['stock_level'] = 3

pricing_sim.compute()

print(f"Pricing Decision: {pricing_sim.output['pricing_decision']:.2f}")
market_demand.view()
competitor_price.view()
production_cost.view()
seasonal_factor.view()
stock_level.view()
pricing_decision.view()
