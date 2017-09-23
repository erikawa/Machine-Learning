
# Data Preprocessing

Preprocessamento dos dados para aplicar ao modelo de dados de Machine Learning

## Importar as bibliotecas

O primeiro passo é importar as bibliotecas utilizadas que são:
* numpy;
* matplotlib.pyplot;
* pandas;


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Criando matriz de amostras

O dataset utilizado contém as colunas: {'**Country**', '**Age**', '**Salary**', '**Purchased**'}


```python
np.set_printoptions(threshold=np.nan)
dataset = pd.read_csv('../resource/Data.csv')
```

Agora precisamos separar as amostras independentes e as amostras dependentes.
As amostras independentes são informações que a princípio não tem relações entre sí, por isso as três primeiras colunas serão minhas amostras independentes.


```python
X = dataset.iloc[:, :-1].values
```

A interação das amostras intependentes resultam nas amostras dependentes. Ou seja, representa a última coluna do nosso dataset


```python
y = dataset.iloc[:, 3].values
```

### Dados Faltantes

Para criar um modelo de dados que será analisado pelo código de machine learning é necessário tratar os dados faltantes do dataset.


```python
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, nan],
           ['France', 35.0, 58000.0],
           ['Spain', nan, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)



Como pode ser observado acima alguns dados estão faltando no dataset.
Existem alguns métodos para tratar os dados faltantes:
* Deletar as linhas com os dados faltantes, mas isso afeta o modelo de dados, pois o resultado vai perder a precisão;
* Substituir os dados faltantes com a média dos valores da coluna, esse é um dos métodos mais utilizados;



```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

Depois de substituir os dados faltantes com a média de suas respectivas colunas temos o resultado abaixo:


```python
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, 63777.77777777778],
           ['France', 35.0, 58000.0],
           ['Spain', 38.77777777777778, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)



### Variáveis Categóricas
Variáveis categóricas são dados que podem ser divididos em categorias, como por exemplo, no dataset as colunas **Country** 
e **Purchased** possúem variáveis categóricas:
1. Coluna **Country**: Possui 3 países diferentes (3 categorias);
2. Coluna **Purchased**: Possui 2 opções (S/N) (2 categorias);


```python

```


```python

```
