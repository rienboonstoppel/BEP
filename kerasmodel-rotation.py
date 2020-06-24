import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def get_model_name(k):
    return 'model_'+str(k)+'.h5'
start_time = time.time()
train_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
train_name = 'GNW_10_100'
print('Loading dataset ' + train_name + '...')
train_dataset = np.loadtxt(train_path + 'positions\\' + train_name + '_data-rotated.txt')
allLabels = np.reshape(np.loadtxt(train_path + 'labels\\' + train_name + '_labels-rotated.txt').astype(int),(1000,100))
print('... took %s seconds' % (time.time() - start_time))
print('Total amount of edges = ' + str(np.sum(allLabels)))
sum_positions = np.sum(allLabels,0)
position = 37
train_labels = allLabels[:,position]

neg, pos = np.bincount(train_labels)
ratio = neg/pos

save_dir = '\\saved_models\\' 

model_save_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\saved_models\\'

model = keras.Sequential(
    [
        layers.Dense(8, input_shape = (len(train_dataset[0]), ), activation="relu", name="layer1"),
        layers.Dense(4, activation="relu", name="layer2"),
        layers.Dense(1, activation="relu", name="layer3")
        # layers.Dense(1, activation="relu", name="layer4")
    ]
)

# model.summary()

class_weight = {0: 1.,
                1: ratio
                }
model.compile(
    loss = keras.losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=0, reduction="auto", name="binary_crossentropy"
        ),
    optimizer = keras.optimizers.SGD(learning_rate=0.01),
    metrics = [get_f1],
)

# Create Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = model_save_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                              monitor='get_f1', 
                                              verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
 	# There can be other callbacks, but just showing one because it involves the model name
 	# This saves the best model

history = model.fit(train_dataset, 
                    train_labels, 
                    batch_size = 2, 
                    epochs = 100, 
                    class_weight = class_weight, 
                    validation_split = 0.2,
                    callbacks = callbacks_list,
                    verbose = 2)

scores = model.evaluate(train_dataset, train_labels, verbose=2)
