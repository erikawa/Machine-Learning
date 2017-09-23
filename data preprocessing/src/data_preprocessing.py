# Data Preprocessing

# Importando as libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuração para exibir todo o conteúdo do array
np.set_printoptions(threshold=np.nan)

# Importando o dataset
dataset = pd.read_csv('../dataset/Data.csv')

# Matriz de amostras
# Amostras Independentes
X = dataset.iloc[:, :-1].values
# Amostras Dependentes
y = dataset.iloc[:, 3].values

# Dados Faltantes
# 1. Deletar as colunas com dados faltantes, porém isso iria afetar o resultado final
# 2. Substituir os dados faltantes com a média dos valores da coluna (Método mais utilizado)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])