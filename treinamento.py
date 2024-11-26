import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados pré-processados
X_train_scaled = pd.read_csv("X_train_scaled.csv", sep=" ", header=None).values
y_train = pd.read_csv("y_train.csv", sep=" ", header=None).values.ravel()
X_test_scaled = pd.read_csv("X_test_scaled.csv", sep=" ", header=None).values
y_test = pd.read_csv("y_test.csv", sep=" ", header=None).values.ravel()

# Definir a arquitetura da Rede Neural com Dropout
model = Sequential([
    Dense(32, input_dim=4, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callback para early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Treinar o modelo
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping]
)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nDesempenho no conjunto de teste:")
print(f"Loss: {loss:.4f}")
print(f"Acurácia: {accuracy:.4f}")

# Fazer previsões
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Relatório e matriz de confusão
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
