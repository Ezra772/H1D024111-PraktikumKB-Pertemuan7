from matplotlib.pyplot import xticks
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Muat dataset iris dari file csv/data
datasets = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, sep=',')

#Menyusun data x (fitur) dan y (label)
x = datasets.iloc[:, :-1].values # 4 kolom pertama sebagai fitur
y = datasets.iloc[:, -1].values # kolom terakhir sebagai label

# Mengonversi label dari string menjadi numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) # Mengubah label jadi 0,1,2

# Memisahkan dataset menjadi data latih dan data validasi rasio 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=x_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test)
)

loss, accuracy = model.evaluate(x_test,y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
pd.DataFrame(history.history).plot(figsize=(10, 6))

pd.DataFrame(history.history).plot(figsize=(10, 6))

predictions = model.predict(x_test)

# Mengambil index dengan nilai tertinggi untuk setiap prediksi 

predicted_classes = predictions.argmax(axis=1)
print("Prediksi: ", predicted_classes)
print("Label Sebenarnya: ", y_test)

#Buat confusion matrix
cm = confusion_matrix(y_test, predicted_classes)

#Visualisasikan confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('predicted')
plt.ylabel('true')
plt.title('Confusion Matrix')
plt.show()

# Fungsi untuk memprediksi data input baru
def predict_new_data():
    sepal_length =float(input("Masukkan sepal length: "))
    sepal_width =float(input("Masukkan sepal width: "))
    petal_length =float(input("Masukkan petal length: "))
    petal_width =float(input("Masukkan petal width: "))

    # Membuat data array baru
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Melakukan prediksi
    prediction = model.predict(new_data)
    predict_class = prediction.argmax(axis=1)

    # Mengonverrsi hasil prediksi kembali ke label asli
    predicted_label = label_encoder.inverse_transform(predict_class)
    print(f"Prediksi kelas:, {predicted_label[0]}")

    predict_new_data()

    
    

