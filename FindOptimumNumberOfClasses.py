import os
import random

import numpy as np
from sklearn.metrics import silhouette_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


class ImageProcessor:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    def load_and_preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)

    def extract_features(self):
        features = []
        filenames = []
        for filename in os.listdir(self.image_directory):
            img_path = os.path.join(self.image_directory, filename)
            processed_image = self.load_and_preprocess_image(img_path)
            features.append(self.model.predict(processed_image).flatten())
            filenames.append(filename)
        return np.array(features), filenames


class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, max_clusters):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_clusters = max_clusters

    def initialize_population(self, num_images):
        return [
            np.random.randint(1, min(i + 2, self.max_clusters + 1), size=num_images)
            for i in range(self.population_size)
        ]

    def fitness(self, individual, features):
        try:
            score = silhouette_score(features, individual)
            return score
        except ValueError:
            return -1

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.randint(1, self.max_clusters + 1)

    def evolve(self, features):
        population = self.initialize_population(features.shape[0])
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind, features) for ind in population]
            sorted_indices = np.argsort(fitness_scores)
            best_individuals = [population[idx] for idx in sorted_indices[-(self.population_size // 2) :]]

            next_generation = best_individuals[:]
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(best_individuals, 2)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.append(child1)
                next_generation.append(child2)

            population = next_generation

        return max(population, key=lambda ind: self.fitness(ind, features))


class ImageClassifier:
    def __init__(self, image_directory, output_file):
        self.processor = ImageProcessor(image_directory)
        self.ga = GeneticAlgorithm(population_size=1000, generations=100000, mutation_rate=0.05, max_clusters=10)
        self.output_file = output_file

    def run(self):
        features, filenames = self.processor.extract_features()
        optimal_classes = self.ga.evolve(features)
        self.output_classification(optimal_classes, filenames)

    def output_classification(self, classes, filenames):
        with open(self.output_file, "w") as file:
            for filename, cluster in zip(filenames, classes):
                file.write(f"{filename}, {cluster}\n")


if __name__ == "__main__":
    classifier = ImageClassifier("path_to_images", "output.txt")
    classifier.run()
