import numpy as np
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

train_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
train_name = 'GNW_50_10'
train_dataset = np.loadtxt(train_path + train_name + '_data.txt')
train_labels = np.loadtxt(train_path + train_name + '_labels.txt').astype(int)
neg, pos = np.bincount(train_labels)
ratio = neg/pos

pred_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
pred_name = 'DREAM_1_100'
pred_dataset = np.loadtxt(pred_path + pred_name + '_data.txt')
pred_labels = np.loadtxt(pred_path + pred_name + '_labels.txt').astype(int)

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

history = model.fit(train_dataset, train_labels, batch_size=16, epochs=10, class_weight=class_weight, validation_split=0.2)

test_scores = model.evaluate(train_dataset, train_labels, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

model.save('model.h5')

predicted_labels = model.predict(pred_dataset)