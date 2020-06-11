from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

pred_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
pred_name = 'DREAM_1_100'
pred_dataset = np.loadtxt(pred_path + pred_name + '_data.txt')
pred_labels = np.loadtxt(pred_path + pred_name + '_labels.txt').astype(int)

model = keras.models.load_model('model.h5',custom_objects={'get_f1':get_f1}) # tensorflow 2.x necessary 
model.summary()

predicted_labels = model.predict_classes(pred_dataset)
predicted_labels = predicted_labels.reshape([100,100])