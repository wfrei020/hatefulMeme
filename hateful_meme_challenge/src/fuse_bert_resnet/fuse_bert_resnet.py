############################################################################################################
######################################### MAIN MULTIMODAL SYSTEM ###########################################
############################################################################################################
# verify GPU availability
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import json
import enum
import math
import random
import shutil
import os
import sys
import matplotlib
matplotlib.use("Agg")
# import the necessary packages

scriptpath = "../resnet/"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))
import resnet152_config as globalconfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet152
import tensorflow_hub as hub
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow import keras

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
    # n_gpu = torch.cuda.device_count()
    # torch.cuda.get_device_name(0)
    return device, n_gpu


def getBertPretrainedModel(bert_model_path, bert_config_path):
    config = BertConfig.from_json_file(bert_config_path)
    bertModel = BertForSequenceClassification(config, num_labels=2)
    state_dict = torch.load(bert_model_path, map_location=torch.device('cpu'))
    bertModel.load_state_dict(state_dict)
    return bertModel


def getResnetPretrainedModel(resnet_model_path):
    resnetModel = keras.models.load_model(resnet_model_path)
    return resnetModel


def buildValidationDirectory():
    with open("../../data/dev.jsonl") as f:
        lines = f.readlines()
        for x in lines:
            if json.loads(x)['label']:
                shutil.copy2(os.path.sep.join([globalconfig.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([globalconfig.VAL_PATH, "hateful"]))
            else:
                shutil.copy2(os.path.sep.join([globalconfig.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([globalconfig.VAL_PATH, "not_hateful"]))


def getTokenizer(type):
    if(type == GET_TOKENIZER_DATA.VALIDATION):
        with open("../../data/dev.jsonl") as f:
            lines = f.readlines()
            labels = [json.loads(x)['label'] for x in lines]
            ids = [json.loads(x)['id'] for x in lines]
            bert_lines = ["[CLS] " + json.loads(x)['text'] + " [SEP]" for x in lines]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in bert_lines]
    # print("Tokenize the first sentence:")
    # print(tokenized_texts[0])

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


def createBertTensors():
    attention_masks, input_ids, labels, ids, nb_labels = getTokenizer(GET_TOKENIZER_DATA.VALIDATION)
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    batch_size = 32

    # Create an iterator of our data with torch DataLoader
    val_data = TensorDataset(train_inputs, train_masks, train_labels)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    return val_data, val_sampler, val_dataloader, ids


def inverse_logits(logits, count, probability_results, label_ids):
    # len(logits)
    for rows in range(len(logits)):
        # print(rows)
        probability_results[count + rows, 0] = math.exp(logits[rows][0]) / (1 + math.exp(logits[rows][0]))
        probability_results[count + rows, 1] = math.exp(logits[rows][1]) / (1 + math.exp(logits[rows][1]))
        probability_results[count + rows, 3] = label_ids[rows]

    return probability_results


def flat_accuracy(preds, labels):
    # there will be len(preds) == 32 for batch, each one containing [probability of 0, probability of 1]
    # here i will need to store this array and comapre the probaility

    # print(inverse_logits(preds[0]))
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def getBertProbabilities(model, validation_data, validation_sampler, validation_dataloader, number_ids):
     # VALIDATION
    probability_results = np.ones([number_ids, 4])  # 4 will have labels
    count = 0
    # Put model in evaluation mode
    model.eval()
    device, num_gpu = specifyDevice()  # i need this to set it to compute
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    # total_predictions = []
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

        probability_results = inverse_logits(logits, count, probability_results, label_ids)
        count = count + 32
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    return probability_results


def getSortArrayofimages(probabilities, results, indexList):
    for i in range(len(probabilities)):
        results[i, 0] = probabilities[i, 0]
        results[i, 1] = probabilities[i, 1]
        results[i, 2] = indexList[i]
    sorted_text_prob_results = results[np.argsort(results[:, 2])]
    return results


############################################################################################################
################################################ PROJECT MAIN #################################################
############################################################################################################
#
#
#
#
#
#
#
def main():
    # handle the text portion
    val_data, val_sampler, val_dataloader, ids = createBertTensors()
    print("Classifying using BERT")
    bertModel = getBertPretrainedModel('../models/hateful_meme_6Epoch.bin', '../models/hateful_meme_config_file_6Epoch.bin')
    probability_results = getBertProbabilities(bertModel, val_data, val_sampler, val_dataloader, len(ids))
    for x in range(len(ids)):
        probability_results[x, 2] = ids[x]

    #print("length of text reuslts")
    # print(len(probability_results))
    sorted_text_prob_results = probability_results[np.argsort(probability_results[:, 2])]
    print("DONE : Classifying using BERT")
    # handle the image portion
    print("Classifying using RESNET152")
    buildValidationDirectory()  # this really only needs to be done once....
    totalTest = len(list(paths.list_images(globalconfig.VAL_PATH)))
    valAug = ImageDataGenerator()
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    valAug.mean = mean
    testGen = valAug.flow_from_directory(
        globalconfig.VAL_PATH,
        class_mode="categorical",
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)
    # print(testGen.filenames[0:500])
    resnetModel = getResnetPretrainedModel('../models/hateful_img_meme.model')
    print("starting validation")
    test_proba = resnetModel.predict(x=testGen, steps=totalTest)
    probability_img_results = np.ones([len(test_proba), 3])
    img_index_list = testingPostProcessing()
    sorted_img_prob_results = getSortArrayofimages(test_proba, probability_img_results, img_index_list)
    # print(sorted_img_prob_results)
    # print("length of image results")
    # print(len(test_proba))
    print("DONE : Classifying using RESNET152")
    correct = 0
    for iter in range(len(sorted_img_prob_results)):
        if sorted_text_prob_results[iter, 0] + sorted_img_prob_results[iter, 0] > sorted_text_prob_results[iter, 1] + sorted_img_prob_results[iter, 1]:
            if 0 == sorted_text_prob_results[iter, 3]:
                correct += 1
        else:
            if 1 == sorted_text_prob_results[iter, 3]:
                correct += 1
    print("Accuracy : ")
    print(correct / len(sorted_img_prob_results))


def testingPostProcessing():
    image_files = []
    image_files_2 = []
    image_files_all = []
    results = np.ones([500, 1])

    for x in os.listdir('../resnet/hateful_nothateful_images/validation/hateful'):
        image_files.append(os.path.splitext(x)[0])
    image_files.sort()
    for x in os.listdir('../resnet/hateful_nothateful_images/validation/not_hateful'):
        image_files_2.append(os.path.splitext(x)[0])
    image_files_2.sort()
    image_files_all = image_files + image_files_2
    for count in range(500):
        results[count] = int(image_files_all[count])

    return results


# testingPostProcessing()

main()
