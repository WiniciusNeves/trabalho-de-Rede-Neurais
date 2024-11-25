import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados pré-processados salvos como CSV
X_train_scaled = pd.read_csv("X_train_scaled.csv").values
y_train = pd.read_csv("y_train.csv").values.ravel()
X_test_scaled = pd.read_csv("X_test_scaled.csv").values
y_test = pd.read_csv("y_test.csv").values.ravel()

# Definir a arquitetura da Rede Neural
model = Sequential([
    Dense(16, input_dim=4, activation='relu'),  # Camada oculta com 16 neurônios
    Dense(8, activation='relu'),               # Camada oculta com 8 neurônios
    Dense(1, activation='sigmoid')             # Camada de saída para classificação binária
])

# Compilar o modelo
model.compile(
    optimizer='adam',                     # Otimizador Adam
    loss='binary_crossentropy',           # Função de perda para classificação binária
    metrics=['accuracy']                  # Métrica de avaliação: acurácia
)

# Treinar a rede neural
history = model.fit(
    X_train_scaled, y_train,              # Dados de treinamento
    validation_split=0.2,                 # 20% do treinamento usado para validação
    epochs=50,                            # Número de épocas
    batch_size=32,                        # Tamanho do lote
    verbose=1                             # Mostrar progresso do treinamento
)

# Avaliar o modelo nos dados de teste
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nDesempenho no conjunto de teste:")
print(f"Loss: {loss:.4f}")
print(f"Acurácia: {accuracy:.4f}")

# Fazer previsões no conjunto de teste
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
