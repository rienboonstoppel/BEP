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

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

train_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
train_name = 'GNW_100_5'
train_dataset = np.loadtxt(train_path + train_name + '_data.txt')
train_labels = np.loadtxt(train_path + train_name + '_labels.txt').astype(int)
neg, pos = np.bincount(train_labels)
ratio = neg/pos

acc_per_fold = []
loss_per_fold = []

save_dir = '\\saved_models\\' 
fold_no = 1
seed = 7
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for train_index, val_index in skf.split(train_dataset, train_labels):
    
    print('Fold: ' + str(fold_no))
    

    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Dense(12, input_shape = (len(train_dataset[0]), ), activation="relu", name="layer1"),
            layers.Dense(8, activation="relu", name="layer2"),
            layers.Dense(1, activation="relu", name="layer3"),
        ]
    )
    
    model.summary()
    
    class_weight = {0: 1.,
                    1: ratio,
                    }
    model.compile(
        loss = keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=0, reduction="auto", name="binary_crossentropy"
            ),
        optimizer = keras.optimizers.SGD(learning_rate=0.01),
        metrics = [get_f1],
    )
    
    # Create Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_no),
                                                  monitor=get_f1, 
                                                  verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
 	# There can be other callbacks, but just showing one because it involves the model name
 	# This saves the best model

    history = model.fit(train_dataset[train_index], 
                        train_labels[train_index], 
                        batch_size = 4, 
                        epochs = 10, 
                        class_weight = class_weight, 
                        validation_data = (train_dataset[val_index], train_labels[val_index]),
                        callbacks = callbacks_list,)
    
    scores = model.evaluate(train_dataset, train_labels, verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test accuracy:", test_scores[1])

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')	