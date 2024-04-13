import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Предобработка данных для применения методов машинного обучения


stress_level_data = pd.read_csv('stress_level.csv', sep=';')

# Работа с пропущенными значениями (NaN)
# Заполенение пропусков в столбце mental_health_history
# Если надо удалить NAN в конкретно столбце то ко будет следующем
delete_mental_health_history_NaN = stress_level_data.dropna(subset=['mental_health_history'])

# Удаляет все пропущеные колонки NaN
deleteAllNaN = stress_level_data.dropna()

# Способ 2. Заполнение средним значением с импользованием fillna()
stress_level_data['mental_health_history'] = stress_level_data.fillna(stress_level_data['stress_level'].mean)
stress_level_data.head()

# Способ 3. Заполнение часто встречающее значение

imputer = SimpleImputer(strategy='most_frequent')
stress_level_data[['mental_health_history']] = imputer.fit_transform(stress_level_data[['stress_level_data']])

# Способ 4. Замена константой
imputer = SimpleImputer(strategy='constant', fill_value=1)

# Обнаружение и устранение выбрасов
stress_level_data.describe()

# Построение ящика с усами
stress_level_data[['Название колонки']].boxplot()

# Усечение значений
# Сделаем усечение кол-во записей в конкретно колонки допустим до 10
stress_level_data.loc[stress_level_data['название колонки'] > 10, 'название колонки'] = 10

# Фильтрация колонки свыше 20 и вывести максимальное значение
stress_level_data[stress_level_data['Название колонки'] < 20]['название колонки'].max()

# Сделаем усечение кол-во записей в конкретно колонки допустим до 5
stress_level_data.loc[stress_level_data['название колонки'] > 5, 'название колонки'] = 5

# Масштабирование

scaler = MinMaxScaler(feature_range=(-1, 1))

stress_level_data[['Название колонки']] = scaler.fit_transform(stress_level_data[['Название колонки']])
