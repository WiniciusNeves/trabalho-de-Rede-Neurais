import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset diretamente da URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

# Nomear as colunas com base na descrição do dataset
columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
data = pd.read_csv(url, header=None, names=columns)

# Exibir as primeiras linhas para verificar a importação
print("Primeiras linhas do dataset:")
print(data.head())

# Separar os dados com base na classe
data_class_1 = data[data["Class"] == 1]
data_class_0 = data[data["Class"] == 0]

# Salvar os dados separados em arquivos CSV
data_class_1.to_csv("data_class_1.csv", index=False)
data_class_0.to_csv("data_class_0.csv", index=False)

print("Dados separados e salvos em 'data_class_1.csv' e 'data_class_0.csv'.")

# Garantir a proporção de 70% para treinamento e 30% para teste
X = data.drop("Class", axis=1)
y = data["Class"]

# Dividir os dados em treinamento e teste, mantendo a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar os atributos para o intervalo [0, 1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Combinar X_train_scaled e y_train para reordenar os dados
train_data_combined = np.column_stack((X_train_scaled, y_train.values))
test_data_combined = np.column_stack((X_test_scaled, y_test.values))

# Separar os dados por classe e depois concatenar
train_class_1 = train_data_combined[train_data_combined[:, -1] == 1]
train_class_0 = train_data_combined[train_data_combined[:, -1] == 0]
test_class_1 = test_data_combined[test_data_combined[:, -1] == 1]
test_class_0 = test_data_combined[test_data_combined[:, -1] == 0]

# Concatenar os dados para que os exemplos da classe 1 venham antes dos da classe 0
train_data_sorted = np.vstack((train_class_1, train_class_0))
test_data_sorted = np.vstack((test_class_1, test_class_0))

# Salvar os dados ordenados
np.savetxt("train_data.txt", train_data_sorted, fmt="%.4f", delimiter=" ")
np.savetxt("test_data.txt", test_data_sorted, fmt="%.4f", delimiter=" ")

print("Arquivos 'train_data.txt' e 'test_data.txt' salvos com a classe 1 antes da classe 0.")