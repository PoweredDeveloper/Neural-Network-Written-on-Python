# Custom Neural Network. Done by Istomin Mikhail
# Simple in preceptron

from datetime import datetime
import os, os.path
from typing import Callable
from PIL import Image
import numpy as np
import copy

# Класс Utils, функции, для удобного использования
class Utils:
    def arraycopy(src, srcPos, dest, destPos, length):
        dest[destPos:destPos+length] = copy.deepcopy(src[srcPos:srcPos+length])
    
    # Расклад изображений на градацию серого (Array)
    def disolveImage(filePath: str) -> list[float]:
        image = Image.open(filePath).convert('RGB')
        output: list[float] = []
        for y in range(image.height):
            for x in range(image.width):
                pixel = image.getpixel((x, y))
                brightness = (sum(pixel) / 3) / 255
                output.append(brightness)
        return output

# Класс слоя
class Layer:
    def __init__(self, size: int, nextSize: int) -> None:
        self.size = size
        self.neurons = np.zeros(size)
        self.biases = np.zeros(size)
        self.weights = np.zeros((size, nextSize))

# Основной класс нейронной сети, именно здесь и находится его главная чать.
class NeuralNetwork:
    # Инициализация, заполнение случайными весами и отклонениями
    def __init__(self, learning_rate: float, sigmoid: Callable[[float], float], dsigmoid: Callable[[float], float], sizes: list[int]) -> None:
        self.learning_rate = learning_rate
        self.sigmoid = sigmoid
        self.dsigmoid = dsigmoid
        self.layers: list[Layer] = []

        for i in range(len(sizes)):
            next_size = 0
            if (i < len(sizes) - 1):
                next_size = sizes[i + 1]
            layer = Layer(sizes[i], next_size)
            layer.biases = np.random.rand(sizes[i]) * 2 - 1
            if (next_size):
                layer.weights = np.random.rand(sizes[i], next_size) * 2 - 1

            self.layers.append(layer)

    # Просчет нейронов до фнального слоя
    def feedForward(self, inputs: list[float]) -> list[float]:
        Utils.arraycopy(inputs, 0, self.layers[0].neurons, 0, len(inputs))
        for i in range(1, len(self.layers)):
            layer_previous = self.layers[i - 1]
            layer_current = self.layers[i]

            for j in range(layer_current.size):
                layer_current.neurons[j] = 0
                for k in range(layer_previous.size):
                    layer_current.neurons[j] += layer_previous.neurons[k] * layer_previous.weights[k][j]

                layer_current.neurons[j] += layer_current.biases[j]
                layer_current.neurons[j] = self.sigmoid(layer_current.neurons[j])
        return self.layers[len(self.layers) - 1].neurons

    # Функция обратной распростронения ошибки
    def backProgation(self, targets: list[float]) -> None:
        errors: list[float] = np.zeros(self.layers[len(self.layers) - 1].size)

        for i in range(self.layers[len(self.layers) - 1].size):
            errors[i] = targets[i] - self.layers[len(self.layers) - 1].neurons[i]

        k = len(self.layers) - 2
        while k >= 0:
            layer_current = self.layers[k]
            layer_next = self.layers[k + 1]
            errors_next: list[float] = np.zeros(layer_current.size)
            gradients: list[float] = np.zeros(layer_next.size)

            for i in range(layer_next.size):
                gradients[i] = errors[i] * self.dsigmoid(self.layers[k + 1].neurons[i])
                gradients[i] *= self.learning_rate
            deltas: list[list[float]] = np.zeros((layer_next.size, layer_current.size))

            for i in range(layer_next.size):
                for j in range(layer_current.size):
                    deltas[i][j] = gradients[i] * layer_current.neurons[j]

            for i in range(layer_current.size):
                errors_next[i] = 0
                for j in range(layer_next.size):
                    errors_next[i] += layer_current.weights[i][j] * errors[j]

            errors: list[float] = np.zeros(layer_current.size)
            Utils.arraycopy(errors_next, 0, errors, 0, layer_current.size)
            weightsNew: list[list[float]] = np.zeros((len(layer_current.weights), len(layer_current.weights[0])))

            for i in range(layer_next.size):
                for j in range(layer_current.size):
                    weightsNew[j][i] = layer_current.weights[j][i] + deltas[i][j]

            layer_current.weights = weightsNew
            for i in range(layer_next.size):
                layer_next.biases[i] += gradients[i]
            k -= 1

# класс обучения нейронной сети
class TrainNetwork:
    def __init__(self, nn: NeuralNetwork, imageFolderPath: str, saveFilePath: str, batchSize: int, epochs: int = 100) -> None:
        self.nn = nn
        self.saveFilePath = saveFilePath
        self.epochs = epochs
        self.imageFolderPath = imageFolderPath
        self.batchSize = batchSize

    # Начать обучение сети
    def Start(self) -> None:
        amountOfFiles = len(os.listdir(self.imageFolderPath)) / self.batchSize if self.epochs == -1 else self.epochs
        imageFiles = os.listdir(self.imageFolderPath)

        for i in range(amountOfFiles):
            current_file: int = 0
            for j in range(self.batchSize):
                fileName: str = imageFiles[current_file]
                targets: list[float] = [-0.1] * 10
                correctValue: int = int(fileName[10])
                targets[correctValue] = 5
                inputNeurons: list[float] = Utils.disolveImage(f'./train/{fileName}')
                forwardResults: list[float] = self.nn.feedForward(inputNeurons)
                targets *= forwardResults
                self.nn.backProgation(targets)
                current_file += 1
            print(f'epoch: {i}.')

start = datetime.now()

sigmoidFunc = lambda x: (1 / (1 + (np.e ** -x)))
dsigmoidFunc = lambda y: (y * (1 - y))

# Создание экземпляров
neuralNetwork = NeuralNetwork(0.001, sigmoidFunc, dsigmoidFunc, [784, 512, 128, 32, 10])
trainNetwork = TrainNetwork(neuralNetwork, 'C:/Users/Miki/Desktop/Neural Network/train', './out', 10, 10)

# Обучение
trainNetwork.Start()
# Тест
results = neuralNetwork.feedForward(Utils.disolveImage('./train/024412-num7.png'))
print('\n'.join([f'{i}: {results[i]}' for i in range(len(results))]))
print('Answer: ', max(results))

print("\nTime: ", datetime.now() - start)