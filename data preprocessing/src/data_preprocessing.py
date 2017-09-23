# Data Preprocessing

# Importando as libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuração para exibir todo o conteúdo do array
np.set_printoptions(threshold=np.nan)

# Importando o dataset
dataset = pd.read_csv('../resource/Data.csv')

# Matriz de amostras
# Amostras Independentes
X = dataset.iloc[:, :-1].values
# Amostras Dependentes
y = dataset.iloc[:, 3].values

# Dados Faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Variáveis categóricas - Trocar texto por números
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
X[:, 0] = labelencoder_y.fit_transform(y)
# Dessa forma o modelo irá entender que existe uma comparação entra as categorias
# sendo que não é isso que queremos
# Variáveis Dummys: Cria uma coluna para cada tipo diferente de categoria

