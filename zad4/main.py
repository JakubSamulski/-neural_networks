import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
#fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

y = np.where(y >= 1, 1, y)
mean_value_thal = X['thal'].mean()
X['thal'].fillna(value=mean_value_thal, inplace=True)
mean_value_ca = X['ca'].mean()
X['ca'].fillna(value=mean_value_ca, inplace=True)
scaler = MinMaxScaler()

# Dopasowanie i transformacja danych (np. macierzy cech X)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(10, input_shape=(13,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=10)

y_pred = (model.predict(X_test) > 0.5).astype(int)

print(accuracy_score(y_test, y_pred))
