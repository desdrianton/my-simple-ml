from kmeans import kmeans


def generate_dummy_data():
    return [
        [1, 1, 1],  # 0
        [4, 3, 5],  # 1
        [5, 7, 6],  # 2
        [8, 2, 0],  # 3
        [1, 3, 9],  # 4
        [0, 1, 2],  # 5
        [9, 4, 7],  # 6
        [8, 5, 5],  # 7
        [6, 9, 4],  # 8
        [6, 0, 7],  # 9
    ]

# ======================================================================================================================
# Running The Models
# ======================================================================================================================
X = generate_dummy_data()
number_of_iteration = 10
centroids_initial = [X[0], X[1]]
kmeans(X, centroids_initial, number_of_iteration)
