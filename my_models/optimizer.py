class Optimizer: # Класс оптимизаторов
    
    def __init__(self, method = 'sgd', alpha=0.9, beta=0.99): # Инициализация объекта класса с заданным видом оптимизатора
        self.method = method
        self.alpha= alpha
        self.beta = beta
        
    def start(self, W, learning_rate, grW): # Инициализация процесса оптимизации для оптимизаторов, с накоплениями
        
        if self.method == 'sgd': 
            return W - learning_rate * grW
        
        elif self.method == 'momentum': # Инициалазиация метода моментов
            self.velocity = grW # Создание памяти
            return W - learning_rate * grW
        
        elif self.method == 'RMSProp': # Инициализация метода Адаптивного градиентного спуска
            self.square_of_grW = grW**2
            self.velocity = (1-self.beta) * self.square_of_grW
            return W - learning_rate * grW / (self.velocity + 10**(-8))**0.5
        
        elif self.method == 'AdaGrad':
            self.square_of_grW = grW**2
            self.velocity = self.square_of_grW
            return W - learning_rate * grW / (self.velocity + 10**(-8))**0.5
        
        elif self.method == 'Adam':
            self.square_of_grW = grW**2
            self.momentum = (1 - self.beta) * self.square_of_grW
            self.velocity = (1-self.alpha) * grW
            m = self.momentum / (1-self.beta)
            v = self.velocity / (1-self.alpha)
            return W - learning_rate * v / (m + 10**(-8))**0.5
            
    def optimize(self, W, learning_rate, grW): # Процесс оптимизации
        
        if self.method == 'sgd':
            return W - learning_rate * grW
        
        elif self.method == 'momentum':
            self.velocity = self.alpha * self.velocity + (1 - self.alpha) * grW # Обновление памяти с коэф. Экс. Сглаживания
            return W - learning_rate * self.velocity
        
        elif self.method == 'RMSProp':
            self.square_of_grW = grW**2
            self.velocity = self.beta * self.velocity  + (1 - self.beta) * self.square_of_grW
            return W - learning_rate * grW / (self.velocity + 10**(-8))**0.5
        
        elif self.method == 'AdaGrad':
            self.square_of_grW = grW**2
            self.velocity = self.velocity  + self.square_of_grW
            return W - learning_rate * grW / (self.velocity + 10**(-8))**0.5
        
        elif self.method == 'Adam':
            self.square_of_grW = grW**2
            self.momentum = self.beta * self.momentum + (1 - self.beta) * self.square_of_grW
            self.velocity = self.alpha * self.velocity + (1 - self.alpha) * grW # Обновление памяти с коэф. Экс. Сглаживания
            m = self.momentum / (1-self.beta)
            v = self.velocity / (1-self.alpha)
            return W - learning_rate * v / (m + 10**(-8))**0.5