import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

heart_rate = ctrl.Antecedent(np.arange(40, 181, 1), 'heart_rate')
blood_pressure = ctrl.Antecedent(np.arange(60, 181, 1), 'blood_pressure')
body_temp = ctrl.Antecedent(np.arange(34, 41, 0.1), 'body_temp')
oxygen_level = ctrl.Antecedent(np.arange(70, 101, 1), 'oxygen_level')
blood_sugar = ctrl.Antecedent(np.arange(50, 201, 1), 'blood_sugar')
insulin = ctrl.Consequent(np.arange(0, 101, 1), 'insulin')
oxygen_therapy = ctrl.Consequent(np.arange(0, 101, 1), 'oxygen_therapy')
medication = ctrl.Consequent(np.arange(0, 101, 1), 'medication')

heart_rate.automf(3)
blood_pressure.automf(3)
body_temp.automf(3)
oxygen_level.automf(3)
blood_sugar.automf(3)
insulin.automf(3)
oxygen_therapy.automf(3)
medication.automf(3)

rules = [
    ctrl.Rule(heart_rate['poor'], oxygen_therapy['good']),
    ctrl.Rule(heart_rate['average'], oxygen_therapy['average']),
    ctrl.Rule(heart_rate['good'], oxygen_therapy['poor']),
    ctrl.Rule(blood_pressure['poor'], medication['good']),
    ctrl.Rule(blood_pressure['average'], medication['average']),
    ctrl.Rule(blood_pressure['good'], medication['poor']),
    ctrl.Rule(body_temp['poor'], medication['good']),
    ctrl.Rule(oxygen_level['poor'], oxygen_therapy['good']),
    ctrl.Rule(oxygen_level['average'], oxygen_therapy['average']),
    ctrl.Rule(oxygen_level['good'], oxygen_therapy['poor']),
    ctrl.Rule(blood_sugar['poor'], insulin['good']),
    ctrl.Rule(blood_sugar['average'], insulin['average']),
    ctrl.Rule(blood_sugar['good'], insulin['poor'])
]

patient_ctrl = ctrl.ControlSystem(rules)
patient_sim = ctrl.ControlSystemSimulation(patient_ctrl)

patient_sim.input['heart_rate'] = 90
patient_sim.input['blood_pressure'] = 120
patient_sim.input['body_temp'] = 37.0
patient_sim.input['oxygen_level'] = 95
patient_sim.input['blood_sugar'] = 110

patient_sim.compute()

print(f"Oxygen Therapy: {patient_sim.output['oxygen_therapy']:.2f}%")
print(f"Medication Level: {patient_sim.output['medication']:.2f}%")
print(f"Insulin Level: {patient_sim.output['insulin']:.2f}%")
heart_rate.view()
blood_pressure.view()
body_temp.view()
oxygen_level.view()
blood_sugar.view()
insulin.view()
oxygen_therapy.view()
