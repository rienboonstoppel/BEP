from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K
import os

def get_f1(y_true, y_pred):
    '''custom metric for keras'''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def loadData(data_path, labels_path):
    '''load the data to be predicted'''
    dataset = np.loadtxt(data_path)
    labels = np.loadtxt(labels_path).astype(int)
    return dataset, labels

def loadModel(model_path):
    '''load the model to be predicted by'''
    model = keras.models.load_model(model_path, custom_objects = {'get_f1': get_f1}) # tensorflow 2.x necessary 
    model.summary()
    return model

def prepareEdgelist(size):
    ''' prepare a list of all possible edges, including a list of indices for selfloops in order to delete them later on'''
    edges = []
    for i in range(size):
        for j in range(size):
            source_node = i+1
            target_node = j+1
            edges.append((source_node, target_node))
    edges = np.array(edges)
        
    selfloops = []
    for i in range(size):
        index = i*size + i
        selfloops.append(index)
    return edges, selfloops

def rankedEdgelist(size, model, dataset):
    '''predict the probabilites of all possible edges and list them, and sort that list'''
    predicted_proba = model.predict(dataset)

    edges, selfloops = prepareEdgelist(size)
                
    unsorted_edges = np.hstack((edges, predicted_proba))
    unsorted_edges = np.delete(unsorted_edges,selfloops,axis=0)
    sorted_edges = unsorted_edges[np.argsort(unsorted_edges[:, 2])]
    sorted_edges_desc = np.flip(sorted_edges,axis=0)
    ranked_edgelist = []
    for i in range(len(sorted_edges_desc)):
        ranked_edgelist.append(('G' + str(int(sorted_edges_desc[i][0])), 'G' + str(int(sorted_edges_desc[i][1])), sorted_edges_desc[i][2]))
    return ranked_edgelist

def saveEdgelist(path, edgelist, save_name):
    newpath = path + r'\ranked_edgelists'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    save_path = newpath + '\\' + save_name
    with open(save_path, 'w') as file:
        file.write('\n'.join('%s\t%s\t%.10f' % x for x in edgelist))

def generateEdgelists(path, preprocessing, amount, model_name, size):
    '''
    generate a ranked edgelist of a GRN of choice
    data is located in the usual folder
    all possible models and checkpoints are saved in ...\saved_model\(folder with date-time)
        in order to use a model for prediction here, the model of your choice should be moved to a folder named 'prediction_model' inside 'saved_models which should be made manually the first time'
    '''
    prediction_modelpath = path + r'\saved_models\prediction_model\\'
    model = loadModel(prediction_modelpath + model_name)

    for i in range(amount):
        data_path = path + '\data\\DREAM_' + str(i+1) + '_1_' + str(size) + '_' + preprocessing + '-data.txt'
        labels_path = path + '\data\\DREAM_' + str(i+1) + '_1_' + str(size) + '_' + preprocessing + '-labels.txt'
        dataset, labels = loadData(data_path, labels_path)
        savename = 'DREAM_' + str(i+1) + '_' + preprocessing + '-rankededgelist.tsv'
        ranked_edgelist = rankedEdgelist(size, model, dataset)
        saveEdgelist(path, ranked_edgelist, savename)

preprocessing = 'KO'
path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP'
model_name ='fold-1.hdf5'
amount = 5
size = 100

generateEdgelists(path, preprocessing, amount, model_name, size)