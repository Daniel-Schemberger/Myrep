import pandas as pd

#Objetivo: construir um algoritmo de ML que identifique se um vinho é tinto ou branco

dados = pd.read_csv(r'D:\Python\dataset\wine_dataset.csv')

print(dados.head())

#renomeando a classificação do vinho

dados['style'] = dados['style'].replace('red',0)
dados['style'] = dados['style'].replace('white',1)

#separando variáveis preditoras e variáveis alvo

y = dados['style'] #armazenando a variável alvo na variável y
x = dados.drop('style', axis=1) #armazenando as variáveis preditoras na variável x

#criando conjuntos de dados de treino e teste

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

#criando o modelo

from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier(random_state=0) #colocando a função do modelo em uma variável
modelo.fit(x_treino, y_treino) #aplicar esta função ao conjunto de dados

#resultados
resultado = modelo.score(x_teste, y_teste)
print(f'Precisão do algorítmo: {resultado}')

print('Escolhendo algumas variáveis teste para utilizar o modelo')

print(y_teste[400:403])
print(x_teste[400:403])


previsao = modelo.predict(x_teste[400:403])
print(previsao)

print('Calculando o erro médio absoluto')
from sklearn.metrics import mean_absolute_error
predict = modelo.predict(x_teste) #variável de predição
mae = mean_absolute_error(y_teste, predict)
print(f'Erro absoluto médio = {mae}')

print('Optimizando o modelo com max_leaf_nodes')

def mae_optimizado(max_leaf_nodes, x_teste, x_treino, y_teste,  y_treino):
    modelo_optz = ExtraTreesClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    modelo_optz.fit(x_treino, y_treino)
    predict_optz = modelo_optz.predict(x_teste)
    mae_optz = mean_absolute_error(y_teste, predict_optz)
    return mae_optz

max_leaf_nodes = [100, 150, 200, 250, 300]

for max_leaf_nodes in [100, 150, 200, 250, 300]:
    nmae = mae_optimizado(max_leaf_nodes, x_teste, x_treino, y_teste,  y_treino)
    print(f'Número de leaf nodes = {max_leaf_nodes} \t\t Erro = {nmae}')

print('Modelo optimizado')

modelf = ExtraTreesClassifier(max_leaf_nodes=250, random_state=0)
modelf.fit(x, y)