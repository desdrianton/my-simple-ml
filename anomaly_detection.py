# Anomaly detection using density algortihmn
import math


def calc_vector_mean(data):
    data_count = len(data)
    feature_count = len(data[0])
    vector_mean = []

    for col in range(feature_count):
        total_per_feature = 0
        for row in range(data_count):
            total_per_feature += data[row][col]
        vector_mean.append(total_per_feature / data_count)

    return vector_mean


def calc_vector_varians(data):
    data_count = len(data)
    feature_count = len(data[0])
    mean = calc_vector_mean(data)
    vector_variances = []

    for col in range(feature_count):
        total_per_feature = 0
        for row in range(data_count):
            total_per_feature += (data[row][col] - mean[col]) ** 2
        vector_variances.append(total_per_feature / data_count)

    return vector_variances


def calc_vector_standard_deviation(data):
    data_count = len(data)
    feature_count = len(data[0])
    variances = calc_vector_varians(data)
    vector_standard_deviation = []

    for col in range(feature_count):
        vector_standard_deviation.append(variances[col] ** 0.5)

    return vector_standard_deviation


def calc_vector_normal_distribution(data):
    data_count = len(data)
    feature_count = len(data[0])
    mean = calc_vector_mean(data)
    standard_deviation = calc_vector_standard_deviation(data)
    normal_distribution = []

    for row in range(data_count):
        normal_distribution.append([])
        for col in range(feature_count):
            divider = math.sqrt(2 * math.pi) * standard_deviation[col]
            power = (-(data[row][col] - mean[col]) ** 2) / (2 * standard_deviation[col] ** 2)
            result = math.e ** power / divider
            normal_distribution[row].append(result)

    return normal_distribution


def calc_vector_total_normal_distribution(data):
    data_count = len(data)
    feature_count = len(data[0])
    normal_distribution = calc_vector_normal_distribution(data)
    total_normal_distribution = []

    for row in range(data_count):
        result = 1
        for col in range(feature_count):
            result *= normal_distribution[row][col]
        total_normal_distribution.append(result)

    return total_normal_distribution


def analyze_vector_normal_distribution(population_data, a_data_to_be_analyzed):
    feature_count = len(population_data[0])
    mean = calc_vector_mean(population_data)
    standard_deviation = calc_vector_standard_deviation(population_data)
    normal_distribution = []

    for col in range(feature_count):
        divider = math.sqrt(2 * math.pi) * standard_deviation[col]
        power = (-(a_data_to_be_analyzed[col] - mean[col]) ** 2) / (2 * standard_deviation[col] ** 2)
        result = math.e ** power / divider
        normal_distribution.append(result)

    return normal_distribution


def analyze_total_normal_distribution(population_data, a_data_to_be_analyzed):
    normal_distribution = analyze_vector_normal_distribution(population_data, a_data_to_be_analyzed)
    result = 1
    for col in range(len(normal_distribution)):
        result *= normal_distribution[col]
    return result
