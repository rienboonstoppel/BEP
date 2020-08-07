#create random network
import numpy as np

for i in range(10):
    random = np.random.rand(100,100)
    save_name = 'random_network_' + str(i+1) + '_knockdowns.tsv'
    np.savetxt(save_name, random, fmt='%.7f', delimiter='\t')
    