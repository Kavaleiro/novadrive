import pandas as pd
import numpy as np
import random as python_random
import joblib  # Para salvar os objetos

from sklearn.preprocessing import StandardScaler,LabelEncoder# Faz a normalização e codificação
from sklearn.model_selection import train_test_split # Faz a divisão dos dados de treino e teste
from sklearn.metrics import classification_report, confusion_matrix # Faz a métricas Para avaliar a performace
from sklearn.ensemble import RandomForestClassifier  # Usado para a seleção de atributos
from sklearn.feature_selection import RFE  # Usado para a seleção de atributos
import tensorflow as tf  # cria a rede neural

from utils import *
import const

# reprodutividade(Garante o mesmo resultado)
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

# dados brutos
df = fetch_data_from_db(const.consulta_sql)

# conversão de tipo
df["idade"] = df["idade"].astype(int)
df["valorsolicitado"] = df["valorsolicitado"].astype(float)
df["valortotalbem"] = df["valortotalbem"].astype(float)

# Tratamento de Nulos
substitui_nulos(df)

# Trata Erros de Digitação
profissoes_validas = [
    "Advogado",
    "Arquiteto",
    "Cientista de Dados",
    "Contador",
    "Dentista",
    "Empresário",
    "Engenheiro",
    "Médico",
    "Programador",
]
corrigir_erros_digitacao(df, "profissao", profissoes_validas)

# Trata Outliers(*POSSIBILIDADE DE FAZER OTLIEARS EM OUTRAS COLUNAS)
df = tratar_outliers(df, "tempoprofissao", 0, 70)
df = tratar_outliers(df, "idade", 0, 110)

# Feature Engineering(Criaremos um campo de porporção ValorSolicitado X ValorTotalBem)
df["proporcaosolicitadototal"] = df["valorsolicitado"] / df["valortotalbem"]
df["proporcaosolicitadototal"] = df["proporcaosolicitadototal"].astype(float)

# Dividindo Dados

X = df.drop("classe", axis=1)
y = df["classe"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# Normalização dos Dados
X_test = save_scalers(
    X_test,
    [
        "tempoprofissao",
        "renda",
        "idade",
        "dependentes",
        "valorsolicitado",
        "valortotalbem",
        "proporcaosolicitadototal",
    ],
)

X_train = save_scalers(
    X_train,
    [
        "tempoprofissao",
        "renda",
        "idade",
        "dependentes",
        "valorsolicitado",
        "valortotalbem",
        "proporcaosolicitadototal",
    ],
)

# Codificação(Transforma dados categóricos em numéricos)
mapeamento = {"ruim": 0, "bom": 1}
y_train = np.array([mapeamento[item] for item in y_train])
y_test = np.array([mapeamento[item] for item in y_test])
X_train = save_encoders(
    X_train,
    ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"],
)
X_test = save_encoders(
    X_test,
    ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"],
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# configurando o otimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compilando o modelo
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
# Treinamento do modelo
model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # Usa 20% dos dados para validação
    epochs=500,  # Número máximo de épocas
    batch_size=10,
    verbose=1,
)
model.save("meu_modelo.keras")

# Previsões
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Avaliando o modelo
print("Avaliação do modelo nos Dado de Teste:")
model.evaluate(X_test, y_test)
# Métricas de Classificação
print("\nRelatório de classificação: ")
print(classification_report(y_test, y_pred))

#criando o código de Explicabilidade
# Função de previsão ajustada para o LIME
def model_predict(data_asarray):
    data_asframe = pd.DataFrame(data_asarray, columns=X_train.columns)#Tranforando o conjunto de dados em um DF
    data_asframe = save_scalers(data_asframe,["tempoprofissao","renda","idade","dependentes","valorsolicitado","valortotalbem","proporcaosolicitadototal"])
    data_asframe = save_encoders(data_asframe,["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"])
    predictions = model.predict(data_asframe)#Chamando a previsão do modelo
    return np.hstack((1-predictions, predictions))


import lime 
import lime.lime_tabular
#Cria o explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['ruim', 'bom'], mode='classification')
exp = explainer.explain_instance(X_test.values[1], model_predict, num_features=10)
#gera o html
exp.save_to_file('lime_explanation.html')

print("Imprimindo os recursos  e seus pesos para BOM")
if 1 in exp.local_exp:
    for feature, weight in exp.local_exp[1]:
        print(f"{feature}: {weight}")
print()
print("Acessar os valores das features e seus peso para BOM")
feature_importances = exp.as_list(label=1)#definindo para bom
for feature, weight in feature_importances:
        print(f"{feature}: {weight}")


