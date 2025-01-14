import pandas as pd
import yaml  # Ler o arquivo de configuração
import psycopg2  # Conecta ao DB(Postgrees)
from fuzzywuzzy import process# Usado para fazer o processamento de lógica difusa(Erro de digitação)
from sklearn.preprocessing import StandardScaler,LabelEncoder# 1:Usado para normalizar os dados; 2:Para as variáveis categóricas
import joblib  # para salvar os nosso OBJECT's
import const  # Importando o arquivo criado(const.py)


def fetch_data_from_db(sql_query):  # Busque dados do DB
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        con = psycopg2.connect(
            dbname=config['database_config']['dbname'], 
            user=config['database_config']['user'], 
            password=config['database_config']['password'], 
            host=config['database_config']['host']
        )

        cursor = con.cursor()
        cursor.execute(sql_query)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    finally:  # fazendo a liberação e logo apos fechando-os
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

    return df

# Resolvendo os valores nulos
def substitui_nulos(df):  # criando um função que irá substituir os nulos. Moda para categoricos e mediana par anuméricos
    for coluna in df.columns:
        if df[coluna].dtype == 'object':  # Caso seja um dado tipo OBJECT
            moda = df[coluna].mode()[0]
            #df[coluna].fillna(moda, inplace=True)  # Substituindo os valore nulos pela moda
            df[coluna].fillna(moda)
        else:  # Caso seja um dado numérico
            mediana = df[coluna].median()
            #df[coluna].fillna(mediana, inplace=True)  # Substituindo os valore nulos pela mediana
            df[coluna].fillna(mediana)

# Erros de Entrada(digitação, sistema, etc..)
"""def corrigir_erros_digitacao(df, coluna, lista_valida):
    for i, valor in enumerate(df[coluna]):
        valor_str = str(valor) if pd.notnull(valor) else valor

        if valor_str not in lista_valida and pd.notnull(valor_str):
            correcao = process.extractOne(valor_str, lista_valida)[0]
            df.at[i, coluna] = correcao"""
def corrigir_erros_digitacao(df, coluna, lista_valida):
    # Itera sobre cada valor da coluna
    for i, valor in enumerate(df[coluna]):
        if pd.notnull(valor):  # Verifica se o valor não é NaN
            valor_str = str(valor)
            # Verifica se o valor não está na lista válida
            if valor_str not in lista_valida:
                # Encontra o valor mais próximo da lista de opções válidas
                correcao = process.extractOne(valor_str, lista_valida)
                if correcao:  # Verifica se há uma correção encontrada
                    df.at[i, coluna] = correcao[0]  # Atualiza o valor da célula com a correção

    return df
# Função que vai trata Outliers(idade e tempo de profissão)
def tratar_outliers(df, coluna, minimo, maximo):
    mediana = df[(df[coluna] >= minimo) & (df[coluna] <= maximo)][coluna].median()
    df[coluna] = df[coluna].apply(lambda x: mediana if x < minimo or x > maximo else x)
    return df

# Scalers usado para a normalização dos dados
def save_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        scaler = StandardScaler()
        df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])
        joblib.dump(scaler, f"./objects/scaler{nome_coluna}.joblib")  # Salvando o objetc dentro da pasta selecionada
    return df

# Encoders para a decodificação
def save_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        label_encoder = LabelEncoder()
        df[nome_coluna] = label_encoder.fit_transform(df[nome_coluna])
        joblib.dump(label_encoder, f"./objects/labelencoder{nome_coluna}.joblib")#Errei no Nome LabelEncoder
    
    return df


# Engenharia de Atributos
# df["proporcaosolicitadototal"] = df["valorsolicitado"] / df["valortotalbem"]

def load_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arquivo_scaler = f"./objects/scaler{nome_coluna}.joblib"
        scaler = joblib.load(nome_arquivo_scaler)
        df[nome_coluna] = scaler.transform(df[[nome_coluna]])
    return df

def load_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arquivo_encoders = f"./objects/labelencoder{nome_coluna}.joblib"
        encoder = joblib.load(nome_arquivo_encoders)
        df[nome_coluna] = encoder.transform(df[nome_coluna])
    return df

"""def load_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arquivo_encoders = f"./objects/labelencoder{nome_coluna}.joblib"
        encoder = joblib.load(nome_arquivo_encoders)

        # Garantir que valores desconhecidos sejam tratados
        df[nome_coluna] = df[nome_coluna].apply(lambda x: x if x in encoder.classes_ else "Desconhecido")

        # Adicionar a categoria 'Desconhecido' ao encoder, se necessário
        if "Desconhecido" not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, "Desconhecido")

        # Transformar os valores
        df[nome_coluna] = encoder.transform(df[nome_coluna])

    return df

"""