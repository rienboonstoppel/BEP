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

val_acc = []
val_loss = []

save_dir = '\\saved_models\\' 
fold_var = 1
seed = 7
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, val_index in skf.split(train_dataset, train_labels):
    
    print('Fold: ' + fold_var)
    
    training_data = np.array([])
    training_labels = np.array([])
    validation_data = np.array([])
    validations_labels = np.array([])
    
    for i in train_index:
        training_data = np.append(training_data, train_dataset[i])
        training_labels = np.append(training_labels, train_labels[i])
    for i in train_index:
        validation_data = np.append(training_data, train_dataset[i])
        validation_labels = np.append(training_data, train_dataset[i])

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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var),
                                                 monitor='val_accuracy', 
                                                 verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
	# There can be other callbacks, but just showing one because it involves the model name
	# This saves the best model

    history = model.fit(train_dataset, 
                        train_labels, 
                        batch_size = 8, 
                        epochs = 10, 
                        class_weight = class_weight, 
                        validation_data = (validation_data, validation_labels),
                        callbacks = callbacks_list,)
    
    # test_scores = model.evaluate(train_dataset, train_labels, verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test accuracy:", test_scores[1])
  	# LOAD BEST MODEL to evaluate the performance of the model

    model.load_weights("/saved_models/model_"+str(fold_var)+".h5")
	
    results = model.evaluate(validation_data, validation_labels, verboxe = 2)
    results = dict(zip(model.metrics_names,results))
	
    val_acc.append(results['accuracy'])
    val_loss.append(results['loss'])
	
    fold_var += 1