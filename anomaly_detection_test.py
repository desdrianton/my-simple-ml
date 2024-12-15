import math
import anomaly_detection as ad


def generate_dummy_data():
    return [
        # Data 2 feature: Tinggi Badan, Berat Badan
        [165, 70],
        [170, 75],
        [175, 80],
        [160, 65],
        [180, 55]
    ]


# ======================================================================================================================
# Running The Models
# ======================================================================================================================
X = generate_dummy_data()
test_data = [120, 90]

vector_mean = ad.calc_vector_mean(X)
print("\nVector Mean:")
print(vector_mean)

vector_variances = ad.calc_vector_varians(X)
print("\nVector Variances:")
print(vector_variances)

vector_standard_deviation = ad.calc_vector_standard_deviation(X)
print("\nVector Standard Deviation:")
print(vector_standard_deviation)

vector_normal_distribution = ad.calc_vector_normal_distribution(X)
print("\nVector Normal Distribution:")
print(vector_normal_distribution)

vector_total_normal_distribution = ad.calc_vector_total_normal_distribution(X)
print("\nVector Total Normal Distribution:")
print(vector_total_normal_distribution)

analyzed_normal_distribution_of_test_data = ad.analyze_vector_normal_distribution(X, test_data)
print(f"\nNormal Distribution of Test Data: {test_data}")
print(analyzed_normal_distribution_of_test_data)

analyzed_total_normal_distribution_of_test_data = ad.analyze_total_normal_distribution(X, test_data)
print(f"\nTotal Normal Distribution of Test Data: {test_data}")
print(analyzed_total_normal_distribution_of_test_data)
