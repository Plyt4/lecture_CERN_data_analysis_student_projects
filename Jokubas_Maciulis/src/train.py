import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

df = pd.read_csv("data/processed/electron_dataset.csv")

X = df.drop(columns = ["target"]).to_numpy()

y = df["target"].to_numpy()

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(32),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(1, activation="sigmoid")  
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2
)