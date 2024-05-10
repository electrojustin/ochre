import numpy as np
import tensorflow as tf
import keras

test_data = np.load('kras_dataset/test_dataset.npy')
test_inputs = test_data[:, :-1]
test_outputs = test_data[:, -1:]
amber_train_data = np.load('active_amber.npy')
amber_train_data = np.concatenate((amber_train_data, np.load('inactive_amber.npy')), axis=0)
amber_train_inputs = amber_train_data[:, :-1]
amber_train_outputs = amber_train_data[:, -1:]
ochre_train_data = np.load('active_ochre.npy')
ochre_train_data = np.concatenate((ochre_train_data, np.load('inactive_ochre.npy')), axis=0)
ochre_train_inputs = ochre_train_data[:, :-1]
ochre_train_outputs = ochre_train_data[:, -1:]
print(test_data.shape)
print(amber_train_data.shape)
print(ochre_train_data.shape)

amber_model = keras.Sequential([
  keras.layers.Dense(1, activation='sigmoid')
])
amber_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
amber_model.fit(x=amber_train_inputs, y=amber_train_outputs, epochs=2)

print('Test data on AMBER model')
amber_model.evaluate(x=test_inputs, y=test_outputs)

ochre_model = keras.Sequential([
  keras.layers.Dense(1, activation='sigmoid')
])
ochre_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
ochre_model.fit(x=ochre_train_inputs, y=ochre_train_outputs, epochs=2)

print('Test data on Ochre model')
ochre_model.evaluate(x=test_inputs, y=test_outputs)

print('AMBER data on Ochre model')
ochre_model.evaluate(x=amber_train_inputs, y=amber_train_outputs)
print('Ochre data on AMBER model')
amber_model.evaluate(x=ochre_train_inputs, y=ochre_train_outputs)

test_model = keras.Sequential([
  keras.layers.Dense(1, activation='sigmoid')
])
test_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
test_model.fit(x=test_inputs, y=test_outputs, epochs=10)
print('Test data on test model')
test_model.evaluate(x=test_inputs, y=test_outputs)
#print(ochre_model.layers[0].get_weights()[0])
#print(amber_model.layers[0].get_weights()[0])
