#
#PLEASE SEE JUNYPER NOTEBOOK, THE TRAINING CODE THERE IS COMPLETE
#
# based on https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
# verify GPU availability
############################################################################################################
########################################## BERT TRAINING MODULE ############################################
############################################################################################################

#PLEASE SEE FUSE_BERT_RESNET.PY FOR THE TESTIN GIMPLEMNENTATION

import tensorflow as tf

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../resnet/')
#import resnet152_config as config

# BERT imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from transformers import BertModel, BertConfig
#from transformers import BertTokenizer, BertForPreTraining, BertForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForMaskedLM, BertForPreTraining, BertForSequenceClassification, BertForTokenClassification, BertModel
#from pytorch_pretrained_bert import BertAdam
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import json

# matplotlib inline
import enum
# Using enum class create enumerations

############################################################################################################
################################################ UTILITIES #################################################
############################################################################################################
class GET_TOKENIZER_DATA(enum.Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3


def checkGPU():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


def specifyDevice():
    device = torch.device("cpu")
    n_gpu = 0
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_gpu = torch.cuda.device_count()
    # torch.cuda.get_device_name(0)
    return device, n_gpu

##BERT TOKENIZER FUNCITON
def getTokenizer(type):
    if(type == GET_TOKENIZER_DATA.TRAINING):
        with open("../../data/dev.jsonl") as f:
            lines = f.readlines()
            labels = [json.loads(x)['label'] for x in lines]
            ids = [json.loads(x)['id'] for x in lines]
            bert_lines = ["[CLS] " + json.loads(x)['text'] + " [SEP]" for x in lines]
    elif(type == GET_TOKENIZER_DATA.VALIDATION):
        with open("../../data/dev.jsonl") as f:
            lines = f.readlines()
            labels = [json.loads(x)['label'] for x in lines]
            ids = [json.loads(x)['id'] for x in lines]
            bert_lines = ["[CLS] " + json.loads(x)['text'] + " [SEP]" for x in lines]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in bert_lines]
    print("Tokenize the first sentence:")
    print(tokenized_texts[0])

    # Set the maximum sequence length.
    MAX_LEN = 128
    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks, input_ids, labels, ids, len(labels)

#BERT. INPUT TENSORS
def create_tensors_for_tuning():

    # lets create variables for training testing and validation
    attention_masks, input_ids, labels, ids, nb_labels = getTokenizer(GET_TOKENIZER_DATA.TRAINING)
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)
    batch_size = 32

    # Create an iterator of our data with torch DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    # validation_sampler = SequentialSampler(validation_data)
    # validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    #GET OUT TRAINED MODEL
    model = BertFineTunningTraining(train_data, train_sampler, train_dataloader, nb_labels)

    #BertTesting(model,validation_data, validation_sampler, validation_dataloader);
    # model.save_pretrained('hateful_meme_text.model')
    # saving model
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    output_model_file = "../models/hateful_meme.bin"
    output_config_file = "../models/hateful_meme_config_file.bin"
    output_vocab_file = "../models/my_own_vocab_file.bin"
    model_to_save = model.module if hasattr(model, 'module') else model

    #torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_vocab_file)
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.

##MY IMPLEMENTATION TO PRODICTIONS
def flat_accuracy(preds, labels):
    print("predictions")
    # there will be len(preds) == 32 for batch, each one containing [probability of 0, probability of 1]
    # here i will need to store this array and comapre the probaility
    #
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# def BertFineTunningTraining(train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader, nb_labels):
############################################################################################################
############################################## TRAINING  ###################################################
############################################################################################################
def BertFineTunningTraining(train_data, train_sampler, train_dataloader, nb_labels):

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    # model.cuda()
    # BERT fine-tuning parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=.1)
    iters = 3
    # Function to calculate the accuracy of our predictions vs labels

    # Store our loss and accuracy for plotting
    train_loss_set = []
    # Number of training epochs
    #epochs = 4
    epochs = 1

    # BERT training loop
    for _ in trange(epochs, desc="Epoch"):

        # TRAINING

        # Set our model to training mode
        model.train()
        device, num_gpu = specifyDevice()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # VALIDATION (THIS ISNT USED HERE ANYMORE)

        # # # Put model in evaluation mode
        # model.eval()

        # # Tracking variables
        # eval_loss, eval_accuracy = 0, 0
        # nb_eval_steps, nb_eval_examples = 0, 0
        # # Evaluate data for one epoch
        # for batch in validation_dataloader:
        #     # Add batch to GPU
        #     batch = tuple(t.to(device) for t in batch)
        #     # Unpack the inputs from our dataloader
        #     b_input_ids, b_input_mask, b_labels = batch
        #     # Telling the model not to compute or store gradients, saving memory and speeding up validation
        #     with torch.no_grad():
        #         # Forward pass, calculate logit predictions
        #         logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        #     # Move logits and labels to CPU
        #     logits = logits.detach().cpu().numpy()

        #     label_ids = b_labels.to('cpu').numpy()
        #     tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #     eval_accuracy += tmp_eval_accuracy
        #     nb_eval_steps += 1
        # print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    # print(model.summary())
    # plot training performance
    return model


create_tensors_for_tuning()


# epoch 6 , step 0.01 weight decay
# called : hateful_meme_6Epoch.bin
# Epoch:  17%|█▋        | 1/6 [06:50<34:12, 410.47s/it]Train loss: 1.082865190349127
# Epoch:  33%|███▎      | 2/6 [13:40<27:21, 410.27s/it]Train loss: 0.5484289286055959
# Epoch:  50%|█████     | 3/6 [20:30<20:30, 410.19s/it]Train loss: 0.48339869356469106
# Epoch:  67%|██████▋   | 4/6 [27:20<13:40, 410.16s/it]Train loss: 0.41595172075400677
# Epoch:  83%|████████▎ | 5/6 [34:11<06:50, 410.33s/it]Train loss: 0.3489604638819408
# Epoch: 100%|██████████| 6/6 [41:01<00:00, 410.25s/it]Train loss: 0.3208405798427144
