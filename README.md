# Average Perceptron Dependency Parser

A simple and accurate dependency parser based on Perceptron model, with feature templates taken from McDonald(2005)

## Author
Quy Nguyen

## Required software
* Numpy library

## Train a parsing model
The software requires having a `train.conll06` and `dev.conll06` files. 
* To train the dataset, type the following at the command prompt:
`python main.py --train train.conll06 --dev dev.conll06 --epochs 5 `

## Parse data with a pretrained model
* The command to test the dataset is:
`python main.py --predict --test test.conll06 [--weights weight.npz] [--featmap feat_map_cutoff]`
* The pretrained parameters could be downloaded
[here](https://drive.google.com/drive/folders/1GT4u3yuo5UoybsIVJXydFDA2_fWTXyPQ?usp=share_link)
