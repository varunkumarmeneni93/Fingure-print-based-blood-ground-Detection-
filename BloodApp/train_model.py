import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# Load preprocessed dataset
X = np.load('model/X.npy')
Y = np.load('model/Y.npy')

X = X.astype('float32') / 255.0
Y = to_categorical(Y)

# Shuffle
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# CNN architecture
cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (3, 3), input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Convolution2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dense(y_train.shape[1], activation='softmax'))
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
os.makedirs("model", exist_ok=True)
model_path = "model/cnn_weights.hdf5"

model_checkpoint = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
history = cnn_model.fit(X_train, y_train, batch_size=32, epochs=40, validation_data=(X_test, y_test), callbacks=[model_checkpoint], verbose=1)

# Save history
with open("model/cnn_history.pckl", "wb") as f:
    pickle.dump(history.history, f)

# Evaluate model
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test1, predict) * 100
precision = precision_score(y_test1, predict, average='macro') * 100
recall = recall_score(y_test1, predict, average='macro') * 100
fscore = f1_score(y_test1, predict, average='macro') * 100

print("\n✅ Training Complete!")
print(f"Accuracy  : {accuracy:.2f}%")
print(f"Precision : {precision:.2f}%")
print(f"Recall    : {recall:.2f}%")
print(f"F1 Score  : {fscore:.2f}%")
print("✅ Model saved as model/cnn_weights.hdf5")
