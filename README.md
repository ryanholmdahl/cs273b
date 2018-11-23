# cs273b

Save each feature as NP array where each row is a drug (given in order of sider_train_dataset_drug_list.csv)

### Text Model

(Yao Liu)

#### Models
The models is defined in ```src/model/text_model.py```. 

1. ```RNNEncoder``` is the base model of RNN. 
2. ```TextClassifier``` is the class of text-ADR classification 
model. It have two embedding models (one is glove for general
 words and the other is specified for medical words). After that 
 it is connected to a LSTM, and then connected with a MLP.
  The out is assumed to be logit (before sigmoid), since I use 
  ```BCEWithLogitLoss```.
3. ```TextBaseModel``` can load ```TextClassifier``` except the last layer.
So we can initialized ```TextBaseModel``` trained ```TextClassifier```
, and use it in the ensemble model.

#### Train and save the text model

Firstly run ```text_preprocess.py``` to preprocess the data. 
Then run ```main_text_model.py``` to train the model.

Hyperparameters are defined in the ```main_text_model.py```. (I use
121 drugs in training set as validation set which can be changed.)

The trained models ```TextClassifier``` will be saved in directory
```/data/save/checkpoint```. 
(```checkpoint``` can be changed.) There is two files, 
```checkpoint.pth.tar``` is the model in last epoch and ```model_
best.pth.tar``` is the model with best validation acc.

To load the saved model, do ```load_checkpoint``` in 
```src/utils.py``` to get the checkpoint and the state_dict of model is in
```checkpoint['state_dict']```. To train the 
text model from checkpoint, run ```python main_text_model.py checkpoint_dir```
, where ```checkpoint_dir``` is the directory in ```/data/save``` that
 contains the saved model ```checkpoint.pth.tar```. 
 (By default it should be ```checkpoint```. I suggest after save 
 the model in ```/data/save/checkpoint```, move it to anther 
 directory ```/data/save/some_folder_name``` and next time
 run ```python main_text_model.py some_folder_name``` to avoid overwrite.) 

Currently I saved a model in 
```/data/save/textmodel_unweight_map0.268_acc93```

#### Metrics and loss

There are several metrics that I am printing during training: P(precision),
R(recall), F(F score), mAP. I use sklearn.metric to calculate those. 
For each there is two way to calculate the average
(over different labels): micro (calculate the score over every element in
each label) and macro (simply take the average of scores for each label). 
See the document for sklearn for details. I'm printing the micro version for P, R and F
and both versions for mAP.

During training, I test on the validation set after each epoch and print it.

For the loss, there is a way to add the weight to the BCEWithLogitLoss with
multi labels. It will adjust the imbalance in positive and negative labels.
In our case, the positive label is very sparse and the unweighted version will
turn to have high precision and low recall. It's suggested to use the ratio #pos/#neg
as the weight but I practice I found the variance is too high, since for 
some label the positive sample is too few. For the weight loss, 
I clip it to make it in [0.1, 10]. There is a hyper-parameter to choose
between weighted or unweighted loss.


