
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
np.set_printoptions(threshold=np.nan, formatter={'float': '{: 0.0f}'.format})
dataset = pd.read_csv('../resource/Data.csv')
```

Agora precisamos separar as amostras independentes e as amostras dependentes.
As amostras independentes são informações que a princípio não tem relações entre sí, por isso as três primeiras colunas serão minhas amostras independentes.


```python
X = dataset.iloc[:, :-1].values
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



A interação das amostras intependentes resultam nas amostras dependentes. Ou seja, representa a última coluna do nosso dataset


```python
y = dataset.iloc[:, 3].values
y
```




    array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'], dtype=object)



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



Depois de substituir os dados faltantes com a média de suas respectivas colunas temos o resultado acima

### Variáveis Categóricas
Variáveis categóricas são dados que podem ser divididos em categorias, como por exemplo, no dataset as colunas **Country** 
e **Purchased** possúem variáveis categóricas:
1. Coluna **Country**: Possui 3 países diferentes (3 categorias);
2. Coluna **Purchased**: Possui 2 opções (S/N) (2 categorias);

#### Colunas Independentes
Basicamente, preciso trocas textos por números para que o algorítmo de classificação entenda o modelo.
Para isso o primeiro passo é transformar a coluna **Country** em categorias:


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X
```




    array([[0, 44.0, 72000.0],
           [2, 27.0, 48000.0],
           [1, 30.0, 54000.0],
           [2, 38.0, 61000.0],
           [1, 40.0, 63777.77777777778],
           [0, 35.0, 58000.0],
           [2, 38.77777777777778, 52000.0],
           [0, 48.0, 79000.0],
           [1, 50.0, 83000.0],
           [0, 37.0, 67000.0]], dtype=object)



Como mostra no resultado acima os países foram substituidos por 3 categorias {0, 1 e 2}.
Mas ainda há um **problema**, da maneira como está esse dataset o algoritmo irá entender que existe uma relação de grandeza atrelado a esses números. Para evitar isso temos que criar uma coluna para cada categoria, como mostra no exemplo abaixo:


```python
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
X
```




    array([[ 1,  0,  0,  44,  72000],
           [ 0,  0,  1,  27,  48000],
           [ 0,  1,  0,  30,  54000],
           [ 0,  0,  1,  38,  61000],
           [ 0,  1,  0,  40,  63778],
           [ 1,  0,  0,  35,  58000],
           [ 0,  0,  1,  39,  52000],
           [ 1,  0,  0,  48,  79000],
           [ 0,  1,  0,  50,  83000],
           [ 1,  0,  0,  37,  67000]])



Agora as categorias estão definidas de forma correta, a coluna *Country* se transformou em 3 colunas,
cada uma representando um país.

#### Colunas Dependentes

Agora precisamos fazer o mesmo com a coluna de amostras dependentes, ou a coluna dos resultados


```python
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y
```




    array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int64)



Pronto! Agora a coluna de amostras dependentes estão categorizadas


```python

```


```python

```
