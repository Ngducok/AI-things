import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

blood_sugar = ctrl.Antecedent(np.arange(50, 251, 1), 'blood_sugar')
bmi = ctrl.Antecedent(np.arange(10, 41, 1), 'bmi')
urination = ctrl.Antecedent(np.arange(0, 4, 1), 'urination')
fatigue = ctrl.Antecedent(np.arange(0, 4, 1), 'fatigue')
diabetes_risk = ctrl.Consequent(np.arange(0, 101, 1), 'diabetes_risk')

blood_sugar['low'] = fuzz.trimf(blood_sugar.universe, [50, 80, 120])
blood_sugar['normal'] = fuzz.trimf(blood_sugar.universe, [80, 120, 160])
blood_sugar['high'] = fuzz.trimf(blood_sugar.universe, [120, 160, 200])
blood_sugar['very_high'] = fuzz.trimf(blood_sugar.universe, [160, 200, 250])

bmi['underweight'] = fuzz.trimf(bmi.universe, [10, 15, 18.5])
bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 22, 25])
bmi['overweight'] = fuzz.trimf(bmi.universe, [25, 30, 35])
bmi['obese'] = fuzz.trimf(bmi.universe, [30, 35, 40])

urination['rarely'] = fuzz.trimf(urination.universe, [0, 0, 1])
urination['sometimes'] = fuzz.trimf(urination.universe, [1, 2, 3])
urination['often'] = fuzz.trimf(urination.universe, [2, 3, 3])

fatigue['low'] = fuzz.trimf(fatigue.universe, [0, 0, 1])
fatigue['moderate'] = fuzz.trimf(fatigue.universe, [1, 2, 3])
fatigue['high'] = fuzz.trimf(fatigue.universe, [2, 3, 3])

diabetes_risk['low'] = fuzz.trimf(diabetes_risk.universe, [0, 20, 40])
diabetes_risk['medium'] = fuzz.trimf(diabetes_risk.universe, [30, 50, 70])
diabetes_risk['high'] = fuzz.trimf(diabetes_risk.universe, [60, 80, 100])

rule1 = ctrl.Rule(blood_sugar['very_high'] | (blood_sugar['high'] & urination['often']), diabetes_risk['high'])
rule2 = ctrl.Rule(blood_sugar['high'] & (bmi['obese'] | bmi['overweight']), diabetes_risk['high'])
rule3 = ctrl.Rule(blood_sugar['normal'] & bmi['normal'], diabetes_risk['low'])
rule4 = ctrl.Rule(blood_sugar['low'] & urination['rarely'], diabetes_risk['low'])
rule5 = ctrl.Rule((blood_sugar['normal'] & fatigue['moderate']) | (blood_sugar['high'] & fatigue['high']), diabetes_risk['medium'])

diabetes_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)

def predict_diabetes(blood_sugar_level, bmi_level, urination_freq, fatigue_level):
    diabetes_sim.input['blood_sugar'] = blood_sugar_level
    diabetes_sim.input['bmi'] = bmi_level
    diabetes_sim.input['urination'] = urination_freq
    diabetes_sim.input['fatigue'] = fatigue_level
    diabetes_sim.compute()
    return diabetes_sim.output['diabetes_risk']

result = predict_diabetes(180, 28, 2, 2)
print(f'Nguy cơ mắc tiểu đường: {result:.2f}%')
blood_sugar.view()
bmi.view()
urination.view()
fatigue.view()
diabetes_risk.view()