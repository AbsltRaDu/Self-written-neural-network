

import numpy as np
from my_models.layer import Layer
from my_models.optimizer import Optimizer
           
class _NeuronNetwork():
    
    def __init__(self, layers=((5, 'sigmoid'),
                               (1, 'leaner')),
                 loss_func='mse'):
        
        '''
        layers - список с кол-вом нейронов на каждом слое
        '''
        
        self.func_of_loss = loss_func
        self.layers = layers 
        self._memory_Grad = 0
        
    def _loss_func(self, y, y_pred): # Ф-ия определения ошибки и производной функционала ошибок

        def mse(y, y_pred):
            return ((y - y_pred).T @ (y - y_pred))/y.shape[0]
        
        def mse_derevative(y):
            return -2/self.size_batch * (y - y_pred) # Считаем градиент MSE по выходу последнего слоя
        
        def logloss(y, y_pred):
            return (-y.T @ np.log(y_pred) - (1-y).T @ np.log(1-y_pred))/y.shape[0]

        def logloss_derevative(y):
            return (-y/self.lst_of_layers[-1].OUTPUT) + ((1-y)/(1-y_pred))
        
        if self.func_of_loss == 'mse':
            self.loss_func = mse(y, y_pred)
            self.loss_derevative = mse_derevative(y)
            
        elif self.func_of_loss == 'logloss':
            self.loss_func = logloss(y, y_pred)
            self.loss_derevative = logloss_derevative(y)


    def _initial_activate_network(self, X): # Ф-ия активации нейронки
        layer = Layer(count_of_feature=X.shape[1], count_of_neuron=self.layers[0][0], func=self.layers[0][1]) # Инициализация нейронной сети, создание первого слоя    
        
        layer.activate(X) # Активация 1-го слоя
        lst_of_layers = [] # Создаем список, где буду храниться объекты слоев
        lst_of_layers.append(layer) # Добавляем объект ранее созданного слоя в список слоев
        
        for i in self.layers[1:]:
            layer = Layer(count_of_feature=lst_of_layers[-1].OUTPUT.shape[1], count_of_neuron=i[0], func=i[1]) # Инициализация нового слоя по списку
            layer.activate(lst_of_layers[-1].OUTPUT) # Активируем слой по матрице выходных данных из предыдущего слоя
            lst_of_layers.append(layer) # Добавляем объект ранее созданного слоя в список слоев
        
        self.lst_of_layers = lst_of_layers
    
    def _activate_network(self, X): # Ф-ия активации нейронки с уже инициализированными нейронами
        self.lst_of_layers[0].activate(X) # В первый слой подаем признаки батча
        for i in range(1, len(self.layers)): # прогоняем нейронку по батчу
            self.lst_of_layers[i].activate(self.lst_of_layers[i-1].OUTPUT) # Активируем слой по матрице выходных данных из предыдущего слоя
    
    # Ф-ия обучения
    def fit(self, X, Y, size_batch=500, learning_rate = 10**(-2), eps=10**(-12), count_of_iteration = 10000, optimizer='sgd', alpha=0.1, reporting=False):
        
        self.size_batch = size_batch # Размер батча
        self.learning_rate = learning_rate # Скорость обучения
        self.eps = eps # Точность обучения
        self.optimizer = optimizer
        self.alpha = alpha
        self.count_of_iteration = count_of_iteration
        self.reporting = reporting
        
        # Блок инициализации сети
      
        self._initial_activate_network(X) # Инициализация сети
        
        # Блок обучения сети
        
        self._loss_func(Y, self.lst_of_layers[-1].OUTPUT)
        
        if self.reporting:
            print(self.loss_func) # Отчет о начальном MSE
        
        # Блок обучение для батча
    
        dct_of_optimizer = {} # Создание словаря для сохранения объектов-оптимизаторов для каждого слоя
        count = 0 # Счетчик для инициализации объектов-оптимизаторов
        while True:
            
            ind = np.random.choice(X.shape[0], size=self.size_batch, replace=False)
            # Y_X_batch = Y_X[ind] # Формируем батч из к наблюдений
            y_batch, X_batch = Y[ind], X[ind] # разбиваем батч результатов и признаков
            self.X_batch = X_batch
            self.y_batch = y_batch
            
            last_loss_func = self.loss_func
            
            # Прогоняем нейронку по батчам
            self._activate_network(X_batch)
            
            self._loss_func(y_batch, self.lst_of_layers[-1].OUTPUT) # Расчет ф-ии ошибок + производная
            if self.reporting:
                print(self.loss_func) # Отчет о MSE на итерации
            
            self.lst_grad_x = [] # Создаем списки для хранения градиентов по X и по W
            self.lst_grad_x.append(self.loss_derevative) # Добавляю конечный градиент ошибки в списки
            
            for layer in self.lst_of_layers[::-1]:
                
                if count == 0: # Инициализация объектов-оптимизаторов
                    
                    gr_1 = layer.OUTPUT_derevative * self.lst_grad_x[-1] # Градиент по ф-ии активации
                    gr_w = (gr_1.T @ layer.INPUT) # Градиент MSE по весам 
                    
                    dct_of_optimizer[(layer)] = Optimizer(method=self.optimizer) # Создание объекта оптимизатора для отдельно взятого слоя
                    layer.W = dct_of_optimizer[layer].start(layer.W, self.learning_rate, gr_w / self.size_batch) # Приращение весов по оптимизатору
                    
                    
                    gr_2 = (layer.W.T @ gr_1.T)[1:].T # Градиент MSE по признакам
                    
                else:
                    gr_1 = layer.OUTPUT_derevative * self.lst_grad_x[-1] # Градиент по ф-ии активации
                    gr_w = (gr_1.T @ layer.INPUT) # Градиент MSE по весам 
                    
                    layer.W = dct_of_optimizer[layer].optimize(layer.W, self.learning_rate, gr_w / self.size_batch) # Приращение весов по оптимизатору
                    
                    
                    gr_2 = (layer.W.T @ gr_1.T)[1:].T # Градиент MSE по признакам
            
                self.lst_grad_x.append(gr_2)
            
            count += 1 # Увеличение счетчика, чтобы уйти с этапа инициализации
            
            if np.isnan(self.loss_func): # Иногда ф-ия ошибок уходит в NaN при больших скачках градиентов и уже не возвращается
                self.loss_func = np.inf
                break
            
            if (np.linalg.norm(self.loss_func - last_loss_func) <= self.eps) or count == self.count_of_iteration:
                break
            
    def return_result_epoch(self): # Ф-ия вывода результатов по эпохам
        for key, value in self.dct_of_result.items(): # Прогоняем словарь моделей
            print(f'{key}: {value.loss_func}') # Выводим результат по каждой
            
    # Ф-ия предсказания                
    def predict(self, X):
        self._activate_network(X) # Прогоняем нейронку по поданным данным
        return self.lst_of_layers[-1].OUTPUT