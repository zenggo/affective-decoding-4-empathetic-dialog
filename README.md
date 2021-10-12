Code of paper [Affective Decoding for Empathetic Response Generation](https://arxiv.org/abs/2108.08102)

5 branches for different experiment settings: master(Transfo), adde(AD, AD+DE), adm(AD + multi-task learning), tml(Transfo + multi-task learning), prepend (Transfo + prepending emotion label predicted by fasttext).

## Usage
```
mkdir log
mkdir save
mkdir save/pretrained_lm // download
```
download the pretrained model params (GPT) from [here](https://github.com/openai/finetune-transformer-lm/tree/master/model). Put the files into `save/pretrained_lm`

### Train
Run the command (GPU will be used if available, make sure CUDA is installed):
```
python train.py --save_path save/model
```

### Interact with model
```
git checkout adm
python play.py --model_path save/model --turns 2
```


### Requirements
* PyTorch (version >=1.4)
* tqdm
* sklearn
* spacy (version < 3)
* ftfy
* pandas
* tensorboardX
