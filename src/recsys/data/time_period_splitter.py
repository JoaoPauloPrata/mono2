import pandas as pd

class TimePeriodSpliter:
    def __init__(self, sliding_window_size, step_size, dataset):
        self.sliding_window_size = sliding_window_size
        self.step_size = step_size
        self.dataset = dataset
        self.dataset['date'] = pd.to_datetime(self.dataset['timestamp'], unit='s')
        self.min_date = self.dataset['date'].min()
        self.max_date = self.dataset['date'].max()
       
    def get_window(self, window_number):
       # Calcular a data de início da janela com precisão de dias
        start_date = self.min_date + pd.DateOffset(months=(window_number - 1) * self.step_size)
        # Calcular a data de término da janela
        end_date = start_date + pd.DateOffset(months=self.sliding_window_size)
        
        print(start_date)
        print(end_date)
        # Verificar se a diferença entre end_date e start_date é menor que o tamanho da janela deslizante
        if (end_date > self.max_date):
            return pd.DataFrame()
        # Filtrar o dataset para incluir apenas as entradas dentro da janela
        window_data = self.dataset[(self.dataset['date'] >= start_date) & (self.dataset['date'] < end_date)]
        
        return window_data
    
    def get_partial_data(self, window, start_month, distance_between_start):
        first_window_date = window['date'].min()
        start_date = first_window_date + pd.DateOffset(months=start_month)
        end_date = start_date + pd.DateOffset(months=distance_between_start)
        partial_data = self.dataset[(self.dataset['date'] >= start_date) & (self.dataset['date'] < end_date)]
        return partial_data
    
    def ShowMaxMin(self):
        print(self.max_date)
        print(self.min_date)

    #TODO boxplot para cada cor
    #TODO tirar meses inclomentos
    #TODO contabilizar itens por usuario por porção de tempo
    #TODO quantos usuarios estão em cada janela - em cada periodo de tempo
    