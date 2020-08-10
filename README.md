# hatefulMeme
Hateful Meme Project / Facebook challenge

My implementation to the hateful Meme challenge given by facebook, i decided not to use any of their libraries implementing current state of the art approaches as i wanted to learn from scratch and come up with my own model.  Future implementation will consist of using their built in tools.

## Junyper Notebook
The hateful_meme.ipynb, is a google colab junyper notebook that implemented my traini and saves my two models one for bert along with its config and the other for my RESNET152 model
if you try to run this please make sure your google colab is using GPU mode

You can run this but it might fail do to aquiring the dataset you need, since the dataset is for competitors only the link to aquire them get updated every couple of days

## Testing

github woudl not let me load the models as they were to large in size.
Please note that testing had to be done with validation dataset provided by facebook, their testing dataset has no labels for competition reasons, so resutls provided by this implementation are on validation dataset.


## Inidividual Training 

in this repo you will find 2 important directories.  

./hateful_meme_challenge/src/bert which is the source code for training bert, this is where i started my development until i needed to move to google colab for their GPUs.
./hateful_meme_challenge/src/resnet which is the source code for training RESNET, this is where i started my development until i needed to move to google colab for their GPUs.
./hateful_meme_challenge/src/resnet/hateful_nothateful_images/ This directory is very important, as it holds my dataset for the validation part used by my testing.  The reason its implemented like this is becasue, i am using generators for the image processing part 

Please consult me for any questions about training, my main description on this paper is the fuse_bert_resnet.py

## Fusing Training BERT and RESNET152 Models
./hateful_meme_challenge/src/fuse_bert_resnet.py
This is my main method in getting the accuracy of the two modles combined.
Please note this python source file runs pretrained models of bert and resnet152 that was training using google colab and saved.

This file should be able to execute if you have the models because i have developed it to be able to run anywhere

I was not able to upload my pretrained models because github size does not allow more the 100 MB files.

The way my fusion took place was to take the output probabilities of each model and then taking the max of the added probabilities

# RESULTS (achieved : 60%)

Configuration for my results
BERT:
  model : BertForSequenceClassification
  pre-trained : "bert-base_uncased"
  optimizer: BertAdam
  Epoch: 6
  learning rate : 0.00002
  training dataset : 8500 sentences
  validation dataset: 500 sentences
  testing dataset: not used as I do not have labels for them
  batch size : 32 
  
RESNET152:
  model : ResNet152
  pre-trained : "imagenet"
  loss : binary_crossentropy
  Epoch: 20
  learning rate:0.0001
  training dataset : 8500 images
  validation dataset: 500 images
  testing dataset: not used as I do not have labels for them
  batch size : 32 
  
Results
  
  Running my program on the validation set with these parameters i achieved 0.6 accuracy or 60%.
  
 ### BERT Accuracy : 0.524
 ### ResNet152 Accuracy : 0.516
 ### Fused Accuracy : 0.60
  
  



