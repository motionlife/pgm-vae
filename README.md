# Probabilistic graphical model parameters tying using auto encoder 
stage 1. Train multiple independent auto-encoders in one neural network.

stage 2. Calculate the pseudo log-likelihood based on the training auto encoder.

##RUN
```
$ python run.py --help
usage: run.py [-h] --name NAME --embedding EMBEDDING --dim DIM [--batch BATCH]
              [--epoch EPOCH] [--rate RATE] [--cost COST] [--ema]
              [--decay DECAY] [--seed SEED] [--device DEVICE] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  target dataset name
  --embedding EMBEDDING, -k EMBEDDING
                        embedding dictionary size
  --dim DIM, -d DIM     embedding dimension
  --batch BATCH, -b BATCH
                        training batch size
  --epoch EPOCH, -e EPOCH
                        number of epochs for training
  --rate RATE, -r RATE  learning rate
  --cost COST, -c COST  commitment cost
  --ema, -m             using exponential moving average
  --decay DECAY, -g DECAY
                        EMA decay rate
  --seed SEED, -s SEED  integer for random seed
  --device DEVICE, -u DEVICE
                        which GPU to use, -1 means only use CPU
  --verbose, -v         verbose mode when do model fitting and sampling
```
Author: Hao Xiong (haoxiong@outlook.com)

##Required package:
 
 Tensorflow 2.0 beta1