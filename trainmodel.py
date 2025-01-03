import os
import numpy as np
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load data as it was saved
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"), allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up callbacks for the model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Use len(actions) for the number of output classes

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Change 'categorical_accuracy' to 'accuracy'
model.fit(X_train, y_train, epochs=80, callbacks=[tb_callback, reduce_lr, early_stopping])
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('new_model2.h5')
