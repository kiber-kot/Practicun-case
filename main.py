import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from preprocessing import delete_nan_value_csv, truncation_values, scaler

#Введение в машинное обучение


#Данные: результаты опроса студентов по уровню стресса по 500 вопросам,
# целевой атрибут stress_level - уровень стресса имеется (1) или нет (0)


delete_nan = delete_nan_value_csv('stress_level.csv')

scaler = scaler(truncation_values(delete_nan))
stress_level_data = scaler


# Определение сбалосирована ли выборка
value_counts = stress_level_data['stress_level'].value_counts()
print('\n')
print('Определение сбалосирована ли выборка', value_counts)
# Связи с тем что выборка является не сбалансированой. А именно
# stress_level
# 1    337
# 0    160
# Будет использована в дальнейшем метрика F1

# Отделяем целевой атрибут Y от предиктора X
x = stress_level_data.drop(columns=['stress_level'])

# Отделяем целевой атрибут X от предиктора Y
y = stress_level_data['stress_level']

# Отделяем обучающую и тестовую часть выборки в соотношении 3:1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

print('\nОтделяем обучающую и тестовую часть выборки в соотношении 3:1')
print('Кол-во тренировочной выборки X ', len(x_train))
print('Кол-во тестовая выборки X ', len(x_test))

# Метод опорных векторов
model_svm = SVC()

# Обученеи модели
model_svm.fit(x_train, y_train)

# Делаем предсказание на тестовой выборке с использованием обученной модели
x_test_predict = model_svm.predict(x_test)

pd.DataFrame({'Предсказаные' : x_test_predict, 'Истиные' : y_test})

# Выечляем метрики
# Метрика accuracy
print('\n')
print('Метрика accuracy ', accuracy_score(y_test, x_test_predict))

# Метрика f1_score
print('Метрика f1_score ', f1_score(y_test, x_test_predict))

print('\n')
# Матрца ошибок
print('Матрца ошибок \n', confusion_matrix(y_test, x_test_predict))

# Общий отчет по метрикам
print("\nОбщий отчет по всем метрикам \n", classification_report(y_test, x_test_predict))

