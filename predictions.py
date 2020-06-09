from tensorflow import keras
import numpy as np

pred_path = 'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\data\\'
pred_name = 'DREAM_1_100'
pred_dataset = np.loadtxt(pred_path + pred_name + '_data.txt')
pred_labels = np.loadtxt(pred_path + pred_name + '_labels.txt').astype(int)

model = keras.models.load_model('C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP\\model.h5')