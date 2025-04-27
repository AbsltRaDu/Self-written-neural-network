class min_max_scaller(): # Класс МинМакс нормировки
    
    def __init__(self): # Инициализация 
        pass

    def fit(self, X): # Обучение 
        self.param = X.min(axis=0), X.max(axis=0) # Сохраняем параметры мин и макс
        
    def transform(self, X): # Нормировка по уже существующим параметрам
        return (X - self.param[0]) / (self.param[1] - self.param[0]) # Номрмировка
    
    def revers_transform(self, X):
        return X * (self.param[1] - self.param[0]) + self.param[0]