# Tensorflow Fine-Gained Image Classifier with Densenet, STN and Multi-attention CNN

This is an [Tensorflow](https://www.tensorflow.org/) implementation of Fine-Gained Image Classier base on [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) with [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) and [Multi-attention CNN](https://www.microsoft.com/en-us/research/publication/learning-multi-attention-convolutional-neural-network-fine-grained-image-recognition/). The models are fine-tuned from [DenseNet-Keras Models](https://github.com/flyyufelix/DenseNet-Keras).

The code are largely borrowed from [TensorFlow-Slim Models](https://github.com/tensorflow/models/tree/master/slim).

## Pre-trained Models

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN)

Network|Top-1|Top-5|Checkpoints
:---:|:---:|:---:|:---:
DenseNet 121 (k=32)| 74.91| 92.19| [model](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA)
DenseNet 169 (k=32)| 76.09| 93.14| [model](https://drive.google.com/open?id=0B_fUSpodN0t0TDB5Ti1PeTZMM2c)
DenseNet 161 (k=48)| 77.64| 93.79| [model](https://drive.google.com/open?id=0B_fUSpodN0t0NmZvTnZZa2plaHc)

## Usage
Follow the instruction [TensorFlow-Slim Models](https://github.com/tensorflow/models/tree/master/slim).

### Step-by-step Example of training on flowers dataset.
#### Downloading ans converting flowers dataset

```
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

#### Training a model from scratch.

```
$ DATASET_DIR=/tmp/data/flowers
$ TRAIN_DIR=/tmp/train_logs
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 
```

#### Fine-tuning a model from an existing checkpoint

```
$ DATASET_DIR=/tmp/data/flowers
$ TRAIN_DIR=/tmp/train_logs
$ CHECKPOINT_PATH=/tmp/my_checkpoints/tf-densenet121.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=global_step,densenet121/logits \
    --trainable_scopes=densenet121/logits
```
