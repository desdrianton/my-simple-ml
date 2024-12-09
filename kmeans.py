import math


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid
        self.points = []

    def generate_info(self):
        return f"centroid = {self.centroid}; points = {self.points}"


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


def calc_square_distance(point1, point2):
    feature_count = len(point1)
    distance = 0
    for i in range(feature_count):
        distance += (point1[i] - point2[i]) ** 2
    return distance


def calc_euclidean_distance(point1, point2):
    return math.sqrt(calc_square_distance(point1, point2))


def find_nearest_cluster(point, clusters):
    shortest_distance = float('inf')
    nearest_cluster = None
    for cluster in clusters:
        distance = calc_euclidean_distance(point, cluster.centroid)
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_cluster = cluster
    return nearest_cluster


def assign_point_to_nearest_cluster(point, clusters):
    nearest_cluster = find_nearest_cluster(point, clusters)
    nearest_cluster.points.append(point)


def calculate_new_centroids(clusters):
    new_centroids = []

    for cluster in clusters:
        new_centroid = [0, 0, 0]

        for point in cluster.points:
            for i in range(len(point)):
                new_centroid[i] += point[i]

        for i in range(len(new_centroid)):
            new_centroid[i] /= len(cluster.points)
        new_centroids.append(new_centroid)

    return new_centroids


def calc_cost_function(clusters):
    cost = 0
    for cluster in clusters:
        for point in cluster.points:
            cost += calc_square_distance(point, cluster.centroid)
    return cost


def generate_cluster_from_centroids(centroids):
    clusters = []
    for centroid in centroids:
        clusters.append(Cluster(centroid))
    return clusters


def kmeans(X, initial_centroids, iteration):
    print(f"Initial centroids: {initial_centroids}")
    print(f"X: {X}")
    print(f"Number of iteration: {iteration}")

    clusters = generate_cluster_from_centroids(initial_centroids)

    for i in range(iteration):
        print(f"\nIteration: {i}")

        for point in X:
            assign_point_to_nearest_cluster(point, clusters)

        cost = calc_cost_function(clusters)

        for j in range(len(clusters)):
            print(f"\tCluster {j}: {clusters[j].generate_info()}")

        print(f"\tcost: {cost}")

        new_centroids = calculate_new_centroids(clusters)
        clusters = generate_cluster_from_centroids(new_centroids)

    return clusters


# ================================================================================
# Running The Models
# ================================================================================
X = generate_dummy_data()
number_of_iteration = 10
centroids_initial = [X[0], X[1]]
kmeans(X, centroids_initial, number_of_iteration)
