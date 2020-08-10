# import the necessary packages
import resnet152_config as config
#from imutils import paths
import random
import shutil
import os
import json

# grab the paths to all input images in the original input directory
# and shuffle them
# THIS ONLY BUILDS VALIDATION DATASET FOR LOCAL COMPUTER

with open("../../data/dev.jsonl") as f:
    lines = f.readlines()
    for x in lines:
        if json.loads(x)['label']:
            shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.VAL_PATH, "hateful"]))
        else:
            shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.VAL_PATH, "not_hateful"]))
# with open("../../project/hateful_meme_challenge/data/dev.jsonl") as f:
#     lines = f.readlines()
#     for x in lines:
#         if count == 0:
#             break
#         if count > 30:
#             if json.loads(x)['label']:
#                 shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.TRAIN_PATH, "hateful"]))
#             else:
#                 shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.TRAIN_PATH, "not_hateful"]))
#             count -= 1
#         elif count > 10:
#             if json.loads(x)['label']:
#                 shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.VAL_PATH, "hateful"]))
#             else:
#                 shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.VAL_PATH, "not_hateful"]))
#             count -= 1
#         else:
#             if json.loads(x)['label']:
#                 shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.TEST_PATH, "hateful"]))
#             else:
#                 shutil.copy2(os.path.sep.join([config.ORIG_INPUT_DATASET, json.loads(x)['img']]), os.path.sep.join([config.TEST_PATH, "not_hateful"]))
#             count -= 1


# loop over the datasets
