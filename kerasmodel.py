import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def loadData(path, name):
    print('Loading dataset ' + name + '...')
    start_time = time.time()
    dataset = np.loadtxt(path + name + '-data.txt')
    labels = np.loadtxt(path + name + '-labels.txt').astype(int)
    print('... took %s seconds' % (time.time() - start_time))
    return dataset, labels


path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
train_name = 'GNW-greedy-nonoise_10_100_PD'
test_name = 'DREAM_1_100_PD'

training = loadData(path, train_name)
testing = loadData(path, test_name)

neg, pos = np.bincount(training[1])
ratio = neg/pos

f1_per_fold = []
loss_per_fold = []

save_dir = '\\saved_models\\' 
fold_no = 1
seed = 10
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

model_save_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\saved_models\\'

for train_index, val_index in skf.split(training[0], training[1]):
    
    print('Fold: ' + str(fold_no))
    

    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Dense(64, input_shape = (len(training[0][0]), ), activation="relu", name="layer1"),
            layers.Dense(32, activation="relu", name="layer2"),
            layers.Dense(1, activation="relu", name="layer3"),
        ]
    )
    
    # model.summary()
    
    class_weight = {0: 1.,
                    1: ratio, #ratio
                    }
    model.compile(
        loss = keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=0, reduction="auto", name="binary_crossentropy"
            ),
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics = [get_f1],
    )
    
    # Create Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = model_save_path + 'fold' + str(fold_no) + '-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                  monitor='get_f1', 
                                                  verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
 	# There can be other callbacks, but just showing one because it involves the model name
 	# This saves the best model

    history = model.fit(training[0][train_index], 
                        training[1][train_index], 
                        batch_size = 64, 
                        epochs = 25, 
                        class_weight = class_weight, 
                        validation_data = (training[0][val_index], training[1][val_index]),
                        callbacks = callbacks_list,
                        verbose = 2)
    
    scores = model.evaluate(testing[0], testing[1], verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test accuracy:", test_scores[1])

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    f1_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# Average scores
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(f1_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - F1: {f1_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> F1: {np.mean(f1_per_fold)} (+- {np.std(f1_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')	