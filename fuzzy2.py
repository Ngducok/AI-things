import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl


cau_demand = ctrl.Antecedent(np.arange(0, 11, 1), 'demand')
gia_competitor = ctrl.Antecedent(np.arange(0, 11, 1), 'competitor_price')
chi_phi_san_xuat = ctrl.Antecedent(np.arange(0, 11, 1), 'production_cost')
muc_ton_kho = ctrl.Antecedent(np.arange(0, 11, 1), 'stock_level')


final_price = ctrl.Consequent(np.arange(0, 11, 1), 'final_price')


cau_demand['thap'] = fuzz.trimf(cau_demand.universe, [0, 0, 5])
cau_demand['trung_binh'] = fuzz.trimf(cau_demand.universe, [3, 5, 7])
cau_demand['cao'] = fuzz.trimf(cau_demand.universe, [5, 10, 10])

gia_competitor['thap'] = fuzz.trimf(gia_competitor.universe, [0, 0, 5])
gia_competitor['trung_binh'] = fuzz.trimf(gia_competitor.universe, [3, 5, 7])
gia_competitor['cao'] = fuzz.trimf(gia_competitor.universe, [5, 10, 10])

chi_phi_san_xuat['thap'] = fuzz.trimf(chi_phi_san_xuat.universe, [0, 0, 5])
chi_phi_san_xuat['trung_binh'] = fuzz.trimf(chi_phi_san_xuat.universe, [3, 5, 7])
chi_phi_san_xuat['cao'] = fuzz.trimf(chi_phi_san_xuat.universe, [5, 10, 10])

muc_ton_kho['thap'] = fuzz.trimf(muc_ton_kho.universe, [0, 0, 5])
muc_ton_kho['trung_binh'] = fuzz.trimf(muc_ton_kho.universe, [3, 5, 7])
muc_ton_kho['cao'] = fuzz.trimf(muc_ton_kho.universe, [5, 10, 10])

final_price['rat_thap'] = fuzz.trimf(final_price.universe, [0, 0, 3])
final_price['thap'] = fuzz.trimf(final_price.universe, [2, 4, 6])
final_price['trung_binh'] = fuzz.trimf(final_price.universe, [4, 6, 8])
final_price['cao'] = fuzz.trimf(final_price.universe, [6, 8, 10])
final_price['rat_cao'] = fuzz.trimf(final_price.universe, [7, 10, 10])


rule1 = ctrl.Rule(cau_demand['cao'] & gia_competitor['thap'] & muc_ton_kho['thap'], final_price['rat_cao'])
rule2 = ctrl.Rule(cau_demand['cao'] & chi_phi_san_xuat['cao'], final_price['cao'])
rule3 = ctrl.Rule(cau_demand['trung_binh'] & gia_competitor['trung_binh'], final_price['trung_binh'])
rule4 = ctrl.Rule(cau_demand['thap'] & muc_ton_kho['cao'], final_price['rat_thap'])


pricing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
pricing = ctrl.ControlSystemSimulation(pricing_ctrl)


pricing.input['demand'] = 7
pricing.input['competitor_price'] = 5
pricing.input['production_cost'] = 6
pricing.input['stock_level'] = 4

pricing.compute()
print("Final Price:", pricing.output['final_price'])
cau_demand.view()
gia_competitor.view()
chi_phi_san_xuat.view()
muc_ton_kho.view()
final_price.view()

