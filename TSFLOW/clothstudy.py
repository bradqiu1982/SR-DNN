# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import kerastuner as kt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



train_images = train_images / 255.0
test_images = test_images / 255.0

def creatmodel(hp):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=(28, 28)));

	# Tune the number of units in the first Dense layer
	# Choose an optimal value between 32-512
	hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
	model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'));
	#model.add(tf.keras.layers.Dense(512, activation='relu'));
	#model.add(tf.keras.layers.Dropout(0.2));
	model.add(tf.keras.layers.Dense(10));

	# Tune the learning rate for the optimizer
	# Choose an optimal value from 0.01, 0.001, or 0.0001
	hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

	#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])
	model.compile(keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])
	return model

tuner = kt.Hyperband(creatmodel, objective='val_sparse_categorical_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs=50,validation_split=0.2)

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
val_acc_per_epoch = history.history['val_sparse_categorical_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(train_images, train_labels, epochs=best_epoch)
eval_result = hypermodel.evaluate(test_images, test_labels, verbose=2)

print(eval_result)
#print(test_images[0])



#model = creatmodel()
#model.fit(train_images, train_labels, epochs=24,validation_data=(test_images,test_labels))
#model.save('./cloth_model.h5')

#new_model = tf.keras.models.load_model('./cloth_model.h5')
#res = new_model.evaluate(test_images, test_labels, verbose=2)
#print(res)

#new_model = creatmodel()
#new_model.load_weights('./cloth_model.h5')
#new_model.evaluate(test_images, test_labels, verbose=2)



#checkpoint_path = "training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
#model.fit(train_images, train_labels, epochs=24,validation_data=(test_images,test_labels),callbacks=[cp_callback])
#model = creatmodel()
#model.evaluate(test_images,  test_labels, verbose=2)
#model.load_weights(checkpoint_path)
#model.evaluate(test_images,  test_labels, verbose=2)

