import pandas as pd
import matplotlib.pyplot as plt  # produção de gráficos
import seaborn as sns  # produção de gráficos

import const
from utils import *

df = fetch_data_from_db(
    const.consulta_sql
)  # chama a função que faz a leitura do DB e armazena nesta Instância

# Formatando os dados obtidos do DB para os tipos corretos
df["idade"] = df["idade"].astype(int)
df["valorsolicitado"] = df["valorsolicitado"].astype(float)
df["valortotalbem"] = df["valortotalbem"].astype(float)

# Criando as lista categoricas
variaveis_categoricas = [
    "profissao",
    "tiporesidencia",
    "escolaridade",
    "score",
    "estadocivil",
    "produto",
]

# Criando as listas numéricas
variaveis_numericas = [
    "tempoprofissao",
    "renda",
    "idade",
    "dependentes",
    "valorsolicitado",
    "valortotalbem",
]

# Criando um laço para fazer os gráficos das Variáveis categoricas
# irá ser usado somente o PLT
for coluna in variaveis_categoricas:
    df[coluna].value_counts().plot(kind="bar", figsize=(10, 4))  # gráfico do tipo barra
    plt.title(f"Distribuição de {coluna}")  # Título
    plt.ylabel("Contagem")  # Eixo vertical
    plt.xlabel(coluna)  # Eixo horizontal
    plt.xticks(rotation=45)  # Evita rótulos cortados
    plt.show()

# Criando um laço que fará gráificos das Variáveis numéricas
# Irá ser usado os SNS também
for coluna in variaveis_numericas:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x=coluna)
    plt.title(f"Boxplot de {coluna}")
    plt.show()

    # Criando o histograma
    df[coluna].hist(bins=20, figsize=(10, 4))  # OBS: "bins" é o número de eixos
    plt.title(f"Histograma de {coluna}")
    plt.xlabel(coluna)
    plt.ylabel("Frequência")
    plt.show()

    # Resumo estatístico
    print(f"Resumo estatítico de {coluna}:\n", df[coluna].describe(), "\n")

nulos_por_coluna = df.isnull().sum()
print(nulos_por_coluna)
