# Importando as libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Configuração para exibir todo o conteúdo do array
np.set_printoptions(threshold=np.nan)

# Importando o dataset
dataset = pd.read_csv('../resource/train_mod.csv')

# Matriz de amostras
# Amostras Independentes
X = dataset.iloc[:, :-1].values
# Amostras Dependentes
y = dataset.iloc[:, 3].values