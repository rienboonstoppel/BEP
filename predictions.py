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

def loadData(data_newpath, labels_newpath):
    dataset = np.loadtxt(data_newpath)
    labels = np.loadtxt(labels_newpath).astype(int)
    return dataset, labels

def loadModel(model_newpath):
    model = keras.models.load_model(model_newpath, custom_objects = {'get_f1': get_f1}) # tensorflow 2.x necessary 
    model.summary()
    return model

def prepareEdgelist(size):
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

def saveEdgelist(edgelist, save_name):
    with open(save_name, 'w') as file:
        file.write('\n'.join('%s\t%s\t%.7f' % x for x in edgelist))

def generateEdgelists(amount, model_name, size):
    size = 100
    preprocessing = 'KO'
    model_path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP\saved_models\prediction_model\\'
    model_newpath = model_path + model_name
    model = loadModel(model_newpath)

    path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP\data\norotation\\'
    for i in range(amount):
        data_path = path + 'DREAM_' + str(i+1) + '_1_' + str(size) + '_' + preprocessing + '-data.txt'
        labels_path = path + 'DREAM_' + str(i+1) + '_1_' + str(size) + '_' + preprocessing + '-labels.txt'
        dataset, labels = loadData(data_path, labels_path)
        savename = 'DREAM_' + str(i+1) + '_1_' + str(size) + '_' + preprocessing + '-rankededgelist.tsv'
        ranked_edgelist = rankedEdgelist(size, model, dataset)
        saveEdgelist(ranked_edgelist, savename)

model_name ='fold-1.hdf5'
amount = 5
size = 100
generateEdgelists(amount, model_name, size)