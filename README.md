# BEP

- Use createdataset-columnrotation.py to create datasets necessary for training, source data included. Saves data to not included 'data' folder
- Use kerasmodel-columnrotation.py to train on the previous generated data. Saves trained model to not included 'saved_models' folder. Copy the desired model to 'saved_models/prediction_model'
- Use predictions.py in order to generate a ranked edgelist based on the generated data and trained model
