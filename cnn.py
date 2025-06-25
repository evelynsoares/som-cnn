import numpy as np

class CNN:
    def __init__(self, input_shape, num_filters, filter_size, num_classes):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_classes = num_classes
        
        # Inicializa os filtros da camada convolucional
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size**2)
        
        # Calcula as dimensões da saída da camada de pooling
        self.pool_output_size = (input_shape[0] - filter_size + 1) // 2  # Assumindo pool_size=2 e stride=2
        self.flattened_size = num_filters * (self.pool_output_size ** 2)
        
        # Inicializa os pesos da camada totalmente conectada
        self.weights = np.random.randn(self.flattened_size, num_classes) / 100
        self.bias = np.zeros(num_classes)

    def convolution(self, input_image):
        """
        Realiza a operação de convolução na entrada usando os filtros.

        Parâmetros:
        - input_image (ndarray): Imagem de entrada como matriz bidimensional.

        Retorno:
        - output (ndarray): Mapa de características resultante da convolução com dimensões reduzidas.
        """
        
        filter_size = self.filter_size
        output_size = self.input_shape[0] - filter_size + 1
        output = np.zeros((self.num_filters, output_size, output_size))

        for f in range(self.num_filters):
            curr_filter = self.filters[f]
            for i in range(output_size):
                for j in range(output_size):
                    region = input_image[i:i+filter_size, j:j+filter_size]
                    output[f, i, j] = np.sum(region * curr_filter)
        return output

    def pooling(self, input_feature_map, pool_size=2, stride=2):
        """
        Realiza a operação de pooling max na entrada.

        Parâmetros:
        - input_feature_map (ndarray): Mapa de características gerado pela camada convolucional.
        - pool_size (int): Tamanho da janela de pooling (default=2).
        - stride (int): Passo do pooling (default=2).

        Retorno:
        - output (ndarray): Mapa de características reduzido após a operação de pooling.
        """

        num_filters, h, w = input_feature_map.shape
        output_h = (h - pool_size) // stride + 1
        output_w = (w - pool_size) // stride + 1
        output = np.zeros((num_filters, output_h, output_w))

        for f in range(num_filters):
            for i in range(0, h - pool_size + 1, stride):
                for j in range(0, w - pool_size + 1, stride):
                    region = input_feature_map[f, i:i+pool_size, j:j+pool_size]
                    output[f, i//stride, j//stride] = np.max(region)
        
        return output

    def forward(self, input_image):
        # Passagem pela camada convolucional
        conv_output = self.convolution(input_image)
        
        # Passagem pela camada de pooling
        pooled_output = self.pooling(conv_output)
        
        # Achata a saída para um vetor
        flattened_output = pooled_output.flatten()
        
        # Passagem pela camada totalmente conectada
        logits = np.dot(flattened_output, self.weights) + self.bias
        return np.argmax(logits)  # Retorna a classe prevista
    


np.random.seed(42)
input_image = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
cnn = CNN(input_shape=(3, 3), num_filters=1, filter_size=2, num_classes=10)
cnn.filters = np.random.randn(1, 2, 2)
cnn.weights = np.random.randn(1 * 1, 10) / 100

predicted_class = cnn.forward(input_image)
print("Classe prevista:", predicted_class)
# Classe prevista: 5