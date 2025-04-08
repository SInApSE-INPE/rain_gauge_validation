import numpy as np
from scipy import stats
import pandas as pd

class EvaluationMetrics:
    def __init__(self, obs, pre):
        """
        Initialize the EvaluationMetrics class with observed and predicted values.
        Metrics based on the work of Plath & Lüdecke (2024) - https://doi.org/10.1080/27684520.2024.2317172
        and Lüdecke et al. (2007) - https://doi.org/10.1175/JCLI4253.1
        """
        self.obs = np.array(obs.copy())
        self.pre = np.array(pre.copy())


    def NMAE(self):
        """
        Normalized Mean Absolute Error (nMAE).
        """
        madx = np.mean(np.abs(self.obs - np.mean(self.obs)))  # Desvio absoluto médio de x
        mady = np.mean(np.abs(self.pre - np.mean(self.pre)))  # Desvio absoluto médio de y
        mae = np.mean(np.abs(self.obs - self.pre))  # Cálculo do MAE
        mean_diff = np.abs(np.mean(self.obs) - np.mean(self.pre))  # Diferença absoluta entre as médias
        mae_max = mean_diff + madx + mady  # Cálculo do MAE máximo
        mae_star = mae / mae_max if mae_max != 0 else 0  # Evita divisão por zero
        return mae_star
    
    def NMSE(self):
        """
        Normalized Mean Squared Error (RMSE).
        """
        d = self.pre - self.obs  # Erro de previsão
        d2_mean = np.mean(d ** 2)  # Média dos erros quadráticos
        Sx = np.std(self.obs, ddof=0)  # Desvio-padrão de x (populacional)
        Sy = np.std(self.pre, ddof=0)  # Desvio-padrão de y (populacional)
        mse_star = d2_mean / (d2_mean + (Sx + Sy) ** 2)
        return mse_star
    
    def PSS(self):
        """
        Probability Density Function (PDF) skill score.
        """
        bins=np.arange(0, 100, .1)
        pdf_obs = np.histogram(self.obs, bins=bins, density=False)[0]/len(self.obs)
        pdf_pre = np.histogram(self.pre, bins=bins, density=False)[0]/len(self.pre)
        return np.sum(np.min([pdf_obs ,pdf_pre],axis=0))
    
    def array_metrics(self):
        """
        Compute all metrics and return them as a dictionary.
        """
        return {'NMAE':self.NMAE(), #[0,1]/0
                'NRMSE':self.NMSE(), #[0,1]/0
                'PSS':self.PSS()} #[0,1]/1

 