import numpy as np

class Layer(): # Класс слоя нейронной сети
    
    def __init__(self, count_of_feature, count_of_neuron = 5, func='sigmoid'): 
        self.W = np.random.random(size=(count_of_neuron, count_of_feature+1)) # Генерация случайной матрицы весов на слое с учетом смещения
        self.func = func # Присвоение ф-ии активации
        
    def activate(self, INPUT): # Подача матрицы INPUT строго размерности (кол-во наблюдений, кол-во признаков) 
        self.INPUT_dervative = INPUT # Входные данные без смещения
        self.INPUT = np.hstack((np.ones((INPUT.shape[0], 1)), INPUT)) # К входной матрице добавляем константу сдвига = 1
        self.LIANER_TRANSFORM = np.array(self.INPUT @ self.W.T) # Матрица линейно-преобразованного входа
        
        def sigmoid(x): # ф-ия активации сигмоида
            return 1/(1+np.exp(-x))
        
        def sigmoid_derevative(x): # производная сигмоиды
            return x * (1-x)
        
        def tanh(x): # Ф-ия активации тангенса
            return np.tanh(x)

        def tanh_derivative(x): # Производная тангенса
            return 1 - np.tanh(x)**2
        
        def relu(x): # Ф-ия активации ReLU
            return np.maximum(0, x)

        def relu_derivative(x): # Производная ReLU
            return np.where(x > 0, 1, 0)
        
        if self.func == 'sigmoid':
            self.OUTPUT = sigmoid(self.LIANER_TRANSFORM) # Применение ф-ии активации сигмоиды
            self.OUTPUT_derevative = sigmoid_derevative(self.LIANER_TRANSFORM) # Производная сигмоиды
        
        if self.func == 'tanh':
            self.OUTPUT = tanh(self.LIANER_TRANSFORM) # Применение ф-ии активации сигмоиды
            self.OUTPUT_derevative = tanh_derivative(self.LIANER_TRANSFORM) # Производная сигмоиды
        
        if self.func == 'ReLU':
            self.OUTPUT = relu(self.LIANER_TRANSFORM) # Применение ф-ии активации сигмоиды
            self.OUTPUT_derevative = relu_derivative(self.LIANER_TRANSFORM) # Производная сигмоиды
        
        if self.func == 'lianer':
            self.OUTPUT = self.LIANER_TRANSFORM
            self.OUTPUT_derevative = 1 # Сделано для того, чтобы при расчете сложной производной в обратном порядке не дублировались значения, так они проходятся по списку