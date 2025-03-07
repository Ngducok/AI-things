import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

muc_cau = ctrl.Antecedent(np.arange(0, 11, 1), 'muc_cau')
gia_doi_thu = ctrl.Antecedent(np.arange(0, 11, 1), 'gia_doi_thu')
chi_phi_san_xuat = ctrl.Antecedent(np.arange(0, 11, 1), 'chi_phi_san_xuat')
muc_ton_kho = ctrl.Antecedent(np.arange(0, 11, 1), 'muc_ton_kho')
gia_cuoi_cung = ctrl.Consequent(np.arange(0, 11, 1), 'gia_cuoi_cung')

muc_cau.automf(3)
gia_doi_thu.automf(3)
chi_phi_san_xuat.automf(3)
muc_ton_kho.automf(3)

gia_cuoi_cung['rat_thap'] = fuzz.trimf(gia_cuoi_cung.universe, [0, 0, 2])
gia_cuoi_cung['thap'] = fuzz.trimf(gia_cuoi_cung.universe, [1, 3, 5])
gia_cuoi_cung['trung_binh'] = fuzz.trimf(gia_cuoi_cung.universe, [4, 5, 7])
gia_cuoi_cung['cao'] = fuzz.trimf(gia_cuoi_cung.universe, [6, 8, 9])
gia_cuoi_cung['rat_cao'] = fuzz.trimf(gia_cuoi_cung.universe, [8, 10, 10])

rule1 = ctrl.Rule(muc_cau['good'] & gia_doi_thu['good'] & muc_ton_kho['poor'], gia_cuoi_cung['rat_cao'])
rule2 = ctrl.Rule(muc_cau['good'] & chi_phi_san_xuat['good'] & muc_ton_kho['poor'], gia_cuoi_cung['rat_cao'])
rule3 = ctrl.Rule(muc_cau['average'] & gia_doi_thu['average'] & muc_ton_kho['average'], gia_cuoi_cung['cao'])
rule4 = ctrl.Rule(muc_cau['poor'] & gia_doi_thu['average'] & muc_ton_kho['average'], gia_cuoi_cung['trung_binh'])
rule5 = ctrl.Rule(muc_cau['poor'] & gia_doi_thu['poor'] & muc_ton_kho['good'], gia_cuoi_cung['rat_thap'])

pricing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
pricing = ctrl.ControlSystemSimulation(pricing_ctrl)

pricing.input['muc_cau'] = 7
pricing.input['gia_doi_thu'] = 6
pricing.input['chi_phi_san_xuat'] = 5
pricing.input['muc_ton_kho'] = 4

pricing.compute()

print(pricing.output['gia_cuoi_cung'])
gia_cuoi_cung.view(sim=pricing)
muc_cau.view()
gia_doi_thu.view()
chi_phi_san_xuat.view()
muc_ton_kho.view()
gia_cuoi_cung.view()