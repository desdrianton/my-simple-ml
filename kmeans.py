import math


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid
        self.points = []

    def generate_info(self):
        return f"centroid = {self.centroid}; points = {self.points}"


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
