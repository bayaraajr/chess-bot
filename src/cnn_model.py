import tensorflow as tf
from tensorflow.keras import layers, models

from load_pgn_output_pairs import load_data_from_pgn

X, y = load_data_from_pgn("data/lichess_db_standard_rated_2015-03.pgn", max_games=500)

model = models.Sequential([
    layers.Input(shape=(8, 8, 12)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4096, activation='softmax')  # 64x64 = 4096 possible move indexes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(X,y)
model.fit(X, y, batch_size=64, epochs=5, validation_split=0.1)

model.save("models/chess_move_cnn.h5")

