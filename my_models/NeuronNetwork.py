from my_models._NeuronNetwork import _NeuronNetwork

class NeuronNetwork():
    
    def __init__(self, layers=((5, 'sigmoid'),
                               (1, 'leaner')),
                 loss_func='mse'):
        
        '''
        layers - список с кол-вом нейронов на каждом слое
        '''
        
        self.func_of_loss = loss_func
        self.layers = layers 
        self._memory_Grad = 0
    
    def fit(self, X, Y, size_batch=500, count_epoch = 5, learning_rate = 10**(-2), eps=10**(-12), count_of_iteration = 10000, optimizer='sgd', alpha=0.1, reporting=False):
        
        self.count_epoch = count_epoch
        self.size_batch = size_batch # Размер батча
        self.learning_rate = learning_rate # Скорость обучения
        self.eps = eps # Точность обучения
        self.optimizer = optimizer
        self.alpha = alpha
        self.count_of_iteration = count_of_iteration
        self.reporting = reporting # Отражать изменение log_loss
        
        self._dct_of_models = {}
        
        for epoch in range(self.count_epoch):
            
            model = _NeuronNetwork(layers=self.layers, loss_func=self.func_of_loss) # Инициализируем одну нейронку
            model.fit(X=X, Y=Y, size_batch=self.size_batch, learning_rate = self.learning_rate, eps=self.eps, count_of_iteration = self.count_of_iteration, optimizer=self.optimizer, alpha=self.alpha, reporting=self.reporting)
            print(f'Эпоха {epoch+1} обучена | LogLoss: {model.loss_func}')
            self._dct_of_models[model] = model.loss_func


