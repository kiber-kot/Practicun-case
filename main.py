from machine_learning import svc
from neural_network import mlp
from preprocessing import delete_nan_value_csv, truncation_values, scaler

#Введение в машинное обучение


#Данные: результаты опроса студентов по уровню стресса по 500 вопросам,
# целевой атрибут stress_level - уровень стресса имеется (1) или нет (0)

"""Предобработка данных"""
delete_nan = delete_nan_value_csv('stress_level.csv')
stress_level_data = scaler(truncation_values(delete_nan))

"""Метод нейроной сети MLP"""
mlp(scaler(truncation_values(delete_nan)))

"""Метод машиного обучения опроных векторов SVC"""
svc(scaler(truncation_values(delete_nan)))

