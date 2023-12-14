
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.src.layers import SimpleRNN

max_features = 20000
batch_size = 32

print('Ładowanie danych...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)


maxlen = 10



input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Trenowanie modelu...')
model.fit(input_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.2)

# Ocena modelu
print('Ocena modelu...')
loss, accuracy = model.evaluate(input_test, y_test, batch_size=batch_size)
print(f'Loss: {loss}, Accuracy: {accuracy}')