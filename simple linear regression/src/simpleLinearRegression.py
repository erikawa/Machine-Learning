# Data Preprocessing

# Importando as libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuração para exibir todo o conteúdo do array
np.set_printoptions(threshold=np.nan)

# Importando o dataset
dataset = pd.read_csv('../resource/Salary_Data.csv')

# Matriz de amostras
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Dividindo o dataset em amostras para treino e para testes
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Normalizando as amostras (Feature Scaling)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.transform(X_test);"""

# Aplicando a Regressão Linear ao dataset de treino
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prevendo os resultados
y_pred = regressor.predict(X_test)