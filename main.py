import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset diretamente da URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

# Nomear as colunas com base na descrição do dataset
columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
data = pd.read_csv(url, header=None, names=columns)

# Exibir uma amostra dos dados
print("Primeiras linhas do dataset:")
print(data.head())

# Separar as classes 1 e 0
class_1 = data[data["Class"] == 1]
class_0 = data[data["Class"] == 0]

# Garantir a proporção de 70% para treinamento e 30% para teste em cada classe
train_1, test_1 = train_test_split(class_1, test_size=0.3, random_state=42, stratify=class_1["Class"])
train_0, test_0 = train_test_split(class_0, test_size=0.3, random_state=42, stratify=class_0["Class"])

# Concatenar os conjuntos de treinamento e teste
train_data = pd.concat([train_1, train_0], axis=0).reset_index(drop=True)
test_data = pd.concat([test_1, test_0], axis=0).reset_index(drop=True)

# Separar os atributos (X) e rótulos (y) para treinamento e teste
X_train = train_data.drop("Class", axis=1)
y_train = train_data["Class"]

X_test = test_data.drop("Class", axis=1)
y_test = test_data["Class"]

# Normalizar os atributos para o intervalo [0, 1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converter os arrays escalados para DataFrames para salvar como CSV
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=["Variance", "Skewness", "Curtosis", "Entropy"])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=["Variance", "Skewness", "Curtosis", "Entropy"])
y_train_df = pd.DataFrame(y_train, columns=["Class"])
y_test_df = pd.DataFrame(y_test, columns=["Class"])

# Salvar os dados pré-processados em CSV
X_train_scaled_df.to_csv("X_train_scaled.csv", index=False)
y_train_df.to_csv("y_train.csv", index=False)
X_test_scaled_df.to_csv("X_test_scaled.csv", index=False)
y_test_df.to_csv("y_test.csv", index=False)

print("Arquivos CSV gerados com sucesso!")
