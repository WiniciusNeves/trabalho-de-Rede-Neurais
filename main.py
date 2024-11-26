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

# Garantir a proporção de 70% para treinamento e 30% para teste
X = data.drop("Class", axis=1)
y = data["Class"]

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar os atributos para o intervalo [0, 1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvar os dados combinados (X e y) no formato compatível
train_data = np.column_stack((X_train_scaled, y_train.values))
test_data = np.column_stack((X_test_scaled, y_test.values))

# Salvar os dados com espaço como delimitador e precisão de 4 casas decimais
np.savetxt("train_data.txt", train_data, fmt="%.4f", delimiter=" ")
np.savetxt("test_data.txt", test_data, fmt="%.4f", delimiter=" ")

# Salvar arquivos separados para uso posterior em CSV
pd.DataFrame(X_train_scaled).to_csv("X_train_scaled.csv", index=False, header=False, sep=" ", float_format="%.4f")
pd.DataFrame(y_train).to_csv("y_train.csv", index=False, header=False, sep=" ")
pd.DataFrame(X_test_scaled).to_csv("X_test_scaled.csv", index=False, header=False, sep=" ", float_format="%.4f")
pd.DataFrame(y_test).to_csv("y_test.csv", index=False, header=False, sep=" ")

print("Pré-processamento concluído. Arquivos salvos com sucesso.")
