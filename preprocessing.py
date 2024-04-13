import pandas as pd

from sklearn.preprocessing import MinMaxScaler

array_full_value_data = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache',
                         'blood_pressure', 'sleep_quality', 'breathing_problem',
                         'noise_level', 'living_conditions', 'safety', 'basic_needs', 'academic_performance',
                         'study_load', 'teacher_student_relationship',
                         'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_activities',
                         'bullying', 'stress_level']
array_not_stress_level_value_data = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache',
                                     'blood_pressure', 'sleep_quality', 'breathing_problem',
                                     'noise_level', 'living_conditions', 'safety', 'basic_needs',
                                     'academic_performance', 'study_load', 'teacher_student_relationship',
                                     'future_career_concerns', 'social_support', 'peer_pressure',
                                     'extracurricular_activities', 'bullying']


def delete_nan_value_csv(path):
    """Убираем все записи, где значение пустое"""
    stress_level_data = pd.read_csv(path, sep=';')
    delete_NaN_data = stress_level_data.dropna()
    return delete_NaN_data


stress_level_data = delete_nan_value_csv('stress_level.csv')


def truncation_values(stress_level_data):
    """Усечение значений.
    В файле имеются аномальные значение в колнках noise_level = 1111111 и future_career_concerns = 101010101"""
    # Фильтрация колонки свыше 20 и вывести максимальное значение
    value_noise_level = stress_level_data[stress_level_data['noise_level'] > 20]['noise_level'].max()
    print('Значение в колонке value_noise_level котрое имеет аномалии = ', value_noise_level)
    value_future_career_concerns = stress_level_data[stress_level_data['future_career_concerns'] > 20][
        'future_career_concerns'].max()
    print('Значение в колонке value_future_career_concerns котрое имеет аномалии = ', value_future_career_concerns)

    # Определяем максимальные значение до усечения в колонке noise_level, где имеются анамалии
    result_noise_level = stress_level_data[stress_level_data['noise_level'] < 20]['noise_level'].max()

    # Сделаем усечение кол-во записей в колонки noise_level до result_noise_level = 5
    stress_level_data.loc[stress_level_data['noise_level'] > result_noise_level, 'noise_level'] = result_noise_level

    # Определяем максимальные значение до усечения в колонке future_career_concerns, где имеются анамалии
    result_future_career_concerns = stress_level_data[stress_level_data['future_career_concerns'] < 20][
        'future_career_concerns'].max()

    # Сделаем усечение кол-во записей в колонки future_career_concerns до result_noise_level = 5
    stress_level_data.loc[stress_level_data[
                              'future_career_concerns'] > result_future_career_concerns, 'future_career_concerns'] = result_future_career_concerns
    return stress_level_data


def scaler(stress_level_data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    stress_level_data[array_not_stress_level_value_data] = scaler.fit_transform(stress_level_data[array_not_stress_level_value_data])
    return stress_level_data

