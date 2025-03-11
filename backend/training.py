# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open("C:\\Aura\\backend\\data.pickle", 'rb'))



# print(type(data_dict['data']))  # Check if it's a list, dict, or something else
# print(len(data_dict['data']))   # Check the number of elements
# print([type(item) for item in data_dict['data']])  # Check the types of elements
# print([np.shape(item) for item in data_dict['data']])  # Check the shapes of elements if they are arrays



# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()





#"C:\\Aura\\backend\\data.pickle"


import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
with open("C:\\Aura\\backend\\data.pickle", 'rb') as f:
    data_dict = pickle.load(f)

# Ensure all sequences in 'data' are padded to have a consistent shape
data = data_dict['data']

# Pad the sequences
data_padded = pad_sequences(data, padding='post', dtype='float32')

# Convert labels to numpy array
labels = np.asarray(data_dict['labels'])

# Label encoding for non-numeric labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Now, 'x_train' and 'x_test' are ready for LSTM training
print("Training Data Shape:", x_train.shape)  # Should be (num_train_samples, max_seq_length)
print("Testing Data Shape:", x_test.shape)  # Should be (num_test_samples, max_seq_length)

# Example: Using LSTM model
model = Sequential([
    Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2] if len(x_train.shape) > 2 else 1)),  # Mask padded values
    LSTM(128, return_sequences=False),  # Adjust LSTM units as needed
    Dense(64, activation='relu'),
    Dense(len(set(labels_encoded)), activation='softmax')  # Adjust for your label count
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Predict the class labels on the test set
y_probs = model.predict(x_test)  # Probabilities
y_predict = np.argmax(y_probs, axis=1)  # Convert probabilities to class labels

# Calculate accuracy
score = accuracy_score(y_test, y_predict)  # Ensure y_test is already integer labels
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
model.save('sign_language_model.h5')  # Saves in Keras format

# Optionally save the trained model using pickle if needed
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
