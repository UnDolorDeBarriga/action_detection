from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard

# from main import get_last_directory
import os
import numpy as np
import csv

DATA_PATH = os.path.join('model/data')
SEQUENCE_LENGTH = 25 # 25 frames per sequence    


with open("model/custom_model_label.csv", encoding='utf-8-sig') as f:
            label_map = csv.reader(f)
            label_map = {
                row[0]: index for index, row in enumerate(label_map)
            }
            actions = list(label_map.keys())

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = def_model(X_train, actions)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()


res = model.predict(X_test)

model.save('action.h5')


def def_model(X, actions):
    """
    Define the model architecture.
    Args:
        X (numpy.ndarray): Input data.
        actions (list): List of action labels.
    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(X.shape[1:])))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    return model