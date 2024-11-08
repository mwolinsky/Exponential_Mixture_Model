import numpy as np
import pandas as pd
class ExponentialMixtureModel():

    def __init__(self, k: int, max_iter: int = 100, tol: float = 1e-6):
        #Cantidad de Clusters
        self.k = k
        #Maxima cantidad de iteraciones del algirtmo EM
        self.max_iter = max_iter
        #Tolerancia para comprar log-likelihood
        self.tol = tol
        #Pesos /pi's
        self.weights = None
        #Parametro lambda de la exponencial
        self.lambdas = None

    def _prepare_data(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        '''
        Prepara los datos de entrada para el algoritmo.

        Args:
        X (array-like): Datos de entrada.
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X debe ser numpy.ndarray o un pandas.DataFrame.")

        if isinstance(X, pd.DataFrame):
            return X.values
        elif X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def initialize_parameters(self, X: np.ndarray | pd.DataFrame) -> None:
        '''
        Inicializa los parámetros del modelo.

        Args:
        X (array-like): Datos de entrada.
        '''
        # Preparar los datos
        X = self._prepare_data(X)
        n_samples, n_features = X.shape
        
        # Se asignan los pesos uniformemente
        self.weights = np.full(self.k, 1 / self.k)
        
        # Se inicializan los cuantiles entre valores de 0.2 y 0.8 para cada clase
        quantiles = np.quantile(X, q=np.linspace(0.2, 0.8, self.k), axis=0)
        self.lambdas = 1 / (quantiles + 1e-6)  # 1/ quantiles para "estimar un posible lambda mejor que lo uniforme"
        
    def e_step(self, X: np.ndarray | pd.DataFrame) -> None:
        '''
        Computa el paso E del algoritmo EM.

        Args:
        X (array-like): Datos de entrada.
        '''

        # En este paso se calcula la probabilidad de pertenecer al cluster j dado las observaciones, ponderándolas 
        # por el peso de cada cluster inicializado anteriormente. Es una regla de bayes ponderada con la exponencial como densidad.
        X = self._prepare_data(X)
        n_samples, n_features = X.shape


        self.e_return = np.zeros((n_samples, self.k))
        for k in range(self.k):
            # Calcular la probabilidad de pertenencia de cada observacion a cada cluster
            probs = self.weights[k] * np.prod(self.lambdas[k] * np.exp(-self.lambdas[k] * X), axis=1)

            
            # Evitar ceros en las probabilidades
            probs = np.where(probs < 1e-10, 1e-10, probs)
            self.e_return[:, k] = probs
        # Normalizar las probabilidades para que sumen 1 por observación
        self.e_return = self.e_return / self.e_return.sum(axis=1, keepdims=True)
        if np.any(np.isnan(self.e_return)):
            raise ValueError("Advertencia: NaNs encontrados en la matriz de probabilidad después del paso E.")
        

    def m_step(self, X):
        X = self._prepare_data(X)
        n_samples, n_features = X.shape
        old_lambdas = self.lambdas.copy()
        
        for k in range(self.k):
            resp_sum = np.sum(self.e_return[:, k])
            self.weights[k] = resp_sum / n_samples
            
            for j in range(n_features):
                weighted_sum = np.sum(self.e_return[:, k] * X[:, j])
                if weighted_sum < 1e-10:
                    self.lambdas[k, j] = old_lambdas[k, j]  # Mantener valor anterior
                else:
                    # Limitar el cambio en lambda para evitar convergencia rápida a un solo cluster
                    self.lambdas[k, j] = resp_sum / weighted_sum

    def fit(self, X):
        X = self._prepare_data(X)
        self.initialize_parameters(X)
        log_likelihood_old = -np.inf
        
        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
            # Calcular log-likelihood
            #log_probs = np.sum(np.log(np.sum(self.e_return, axis=1) + 1e-300))
            log_probs=0
            for i in range(X.shape[0]):  # Iteramos sobre cada observación
                total_density = 0
                for k in range(self.k):  # Iteramos sobre cada componente
                    # Calculamos la densidad para el componente k, ponderada por su peso
                    density = self.weights[k] * np.prod(self.lambdas[k] * np.exp(-self.lambdas[k] * X[i]))
                    total_density += density
                # Sumamos el logaritmo de la densidad total de la observación i al log-likelihood
                log_probs += np.log(total_density + 1e-300)  # Evitar log(0)

            # Verificar si los clusters están balanceados
            cluster_weights = np.mean(self.e_return, axis=0)
            max_weight_ratio = np.max(cluster_weights) / np.min(cluster_weights)
            
            if i > 0 and np.abs(log_probs - log_likelihood_old) < self.tol:
                if max_weight_ratio > 100:  # Si los clusters están muy desbalanceados
                    print("Reiniciando debido a clusters desbalanceados...")
                    self.initialize_parameters(X)
                    log_likelihood_old = -np.inf
                    continue
                    
                print(f"Convergencia alcanzada en la iteración {i+1}")
                break
                
            log_likelihood_old = log_probs

    def predict_proba(self, X):
        X = self._prepare_data(X)
        self.e_step(X)
        return self.e_return

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)