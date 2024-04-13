import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def create_mlp(x_train, y_train):
    # Определение гиперпараметров: 2 скрытых слоя по 50 нейронов в каждом
    hidden_layer_sizes = (50, 50)
    # Создание объекта классификатора
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500)
    # Обучение нейронной сети на данных
    return clf.fit(x_train, y_train)


def mlp(stress_level_data):
    print('-----------------------------------------------------------------------------------------------')
    print('Метод нейроная сеть MLP')
    global value_counts
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
    model_svm = create_mlp(x_train, y_train)
    # Делаем предсказание на тестовой выборке с использованием обученной модели
    x_test_predict = model_svm.predict(x_test)
    pd.DataFrame({'Предсказаные': x_test_predict, 'Истиные': y_test})
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
    print('-----------------------------------------------------------------------------------------------')

