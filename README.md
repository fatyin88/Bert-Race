# BERT for RACE (for NLP Project Usage)

By: Chenglei Si (River Valley High School)

### Update:
XLNet has achieved impressive gains on RACE recently. You may refer to my other repo: https://github.com/NoviScl/XLNet_DREAM to see how to use XLNet for multiple-choice machine comprehension problems. Huggingface has updated their work [pytorch_trainsformers](https://github.com/huggingface/pytorch-transformers), please refer to their repo for the documentation and more details of the new version. 

### Note:
Note: You should use the dev set to do hyper-parameter tuning and then use the test file and trained model to evaluate on the test data. This is the standard practice for ML.

### Implementation
This work is based on Pytorch implementation of BERT (https://github.com/huggingface/pytorch-pretrained-BERT). I adapted the original BERT model to work on multiple choice machine comprehension.

### Environment:
The code is tested with Python3.6 and Pytorch 1.0.0.
Have to install Apex first

### Usage
1. 
from google.colab import drive
drive.mount('/content/drive')
2. 
! git clone https://github.com/NVIDIA/apex
cd /content/apex
! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
3. 
cd /content/BERT-RACE
! python run_race.py --data_dir=/content/drive/My\ Drive/COMP\ NLP\ Group\ Project/RACE --bert_model=bert-base-uncased --output_dir=base_models --max_seq_length=450 --do_train --do_eval --do_lower_case --train_batch_size=24 --eval_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3 --gradient_accumulation_steps=8 --fp16 --loss_scale=128 && /root/shutdown.sh

1. Download the dataset and unzip it. The default dataset directory is ./RACE
2. Run ```./run.sh```

### Hyperparameters
I did some tuning and find the following hyperparameters to work reasonally well:

BERT_base: batch size: 32, learning rate: 5e-5, training epoch: 3

BERT_large: batch size: 8, learning rate: 1e-5 (DO NOT SET IT TOO LARGE), training epoch: 2

### Results
Model | RACE | RACE-M | RACE-H 
--- | --- | --- | --- |
BERT_base | 65.0 | 71.7 | 62.3 
BERT_large | 67.9 | 75.6 | 64.7

You can compare them with other results on the [leaderboard](http://www.qizhexie.com/data/RACE_leaderboard).

BERT large achieves the current (Jan 2019) best result. Looking forward to new models that can beat BERT!

### More Details
I have written a short [report](./BERT_RACE.pdf) in this repo describing the details.




