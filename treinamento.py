import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados
X_train_scaled = np.load("X_train_scaled.npy")
y_train = np.load("y_train.npy")
X_test_scaled = np.load("X_test_scaled.npy")
y_test = np.load("y_test.npy")

# Definir a arquitetura da Rede Neural com Dropout
model = Sequential([
    Dense(32, input_dim=4, activation='relu'),  # Camada oculta com 32 neurônios
    Dropout(0.3),                              # Dropout de 30%
    Dense(16, activation='relu'),              # Camada oculta com 16 neurônios
    Dropout(0.2),                              # Dropout de 20%
    Dense(1, activation='sigmoid')             # Camada de saída para classificação binária
])

# Compilar o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Ajustar o learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Definir callbacks para interromper o treinamento se não houver melhora
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,               # Interromper se não houver melhora em 10 épocas
    restore_best_weights=True  # Restaurar os pesos do melhor modelo
)

# Treinar o modelo
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,                 # Aumentar o número de épocas
    batch_size=16,              # Reduzir o batch size
    verbose=1,
    callbacks=[early_stopping]
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

# Matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
