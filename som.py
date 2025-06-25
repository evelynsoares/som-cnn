import numpy as np

class SOM:
    def __init__(self, input_dim, map_size, learning_rate=0.1, sigma=1.0):
        self.input_dim = input_dim
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

    def find_best_matching_unit(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_weights(self, input_vector, bmu_index, epoch, max_epochs):
        lr = self.learning_rate * (1 - epoch / max_epochs)
        radius = self.sigma * (1 - epoch / max_epochs)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                influence = np.exp(-distance_to_bmu**2 / (2 * radius**2))
                self.weights[i, j] += lr * influence * (input_vector - self.weights[i, j])

    def train(self, data, epochs):
        for epoch in range(epochs):
            for input_vector in data:
                bmu_index = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu_index, epoch, epochs)
        
        # Após o treinamento, retornamos a BMU para o primeiro vetor de entrada
        return self.find_best_matching_unit(data[0])

# Teste
data = np.array([[0.5, 0.5], [0.7, 0.7]])  # Dois pontos de dados fixos
som = SOM(input_dim=2, map_size=(5, 5), learning_rate=0.1, sigma=1.0)

np.random.seed(42)
som.weights = np.random.rand(5, 5, 2)  # Pesos fixos

# Treinamento e saída
bmu_position = som.train(data, epochs=10)
print("Posição da BMU:", bmu_position)