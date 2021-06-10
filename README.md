# List of nested NER benchmarks

Nested NER problem on [paperswithcode](https://paperswithcode.com/task/nested-named-entity-recognition).  
General datasets are [GENIA](http://www.geniaproject.org/genia-corpus) and [ACE2005](https://catalog.ldc.upenn.edu/LDC2006T06)
# Dataset
Dataset has 29 categories:  
NUMBER, WORK_OF_ART, PROFESSION, LANGUAGE, PRODUCT, DISEASE, MONEY, NATIONALITY, ORGANIZATION, DATE, AWARD, DISTRICT, FACILITY, AGE, LOCATION, PERSON, STATE_OR_PROVINCE, EVENT, COUNTRY, LAW, PENALTY, FAMILY, TIME, PERCENT, CRIME, IDEOLOGY, ORDINAL, CITY, RELIGION

Category stats per train/dev/test: TBA

To convert dataset into different formats look at [notebooks/dataset_conversion.ipynb](notebooks/dataset_conversion.ipynb)  
Preconverted datasets (from [10.04.2021](https://disk.yandex.ru/d/b7uDk5vxZCi6Ug?w=1)):
* jsonlines for Biaffine NER - [link](https://disk.yandex.ru/d/TOUjOmmSCpPk1A?w=1)  
* json for Pyramid NER - [link](https://disk.yandex.ru/d/GSmLv62NsB2r_Q?w=1)  
* conll-like for seq2seq NER - [link](https://disk.yandex.ru/d/qmJebDNXEDpwNw?w=1)  
* json for mrc-for-flat-nested-ner - [link](https://disk.yandex.ru/d/gzVPejuVPuXcZQ)

# Language models
* FastText  
[cc300ru](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz)
* BERT  
[ruBERT by DeepPavlov](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz)  
[ruBERT cased by DeepPavlov + huggingface](https://huggingface.co/DeepPavlov/rubert-base-cased/tree/main)

# Methods
Main metric is F1-score. The named entity is considered correct when both boundary and
category are predicted correctly.
## Biaffine NER
[link to article](https://arxiv.org/pdf/2005.07150.pdf)  
[link to implementation (tensorflow>=2.2)](https://github.com/jsmith09/biaffine-ner)  
[link to original implementation (tensorflow<2.0 and python 2)](https://github.com/juntaoy/biaffine-ner)
### Training
1. Look through experiment configurations at [experiments.conf](repos/biaffine-ner-tf2.2/experiments.conf)  
2. And run:  
```bash 
cd repos/biaffine-ner
python train.py experiment5
```
### Pretrained models
* char-cnn + fasttext embeddings (cc300ru) - [link](https://disk.yandex.ru/d/Q4O2MVvWppvoUA?w=1)
* char-cnn + fasttext (cc300ru) + ruBERT embeddings - [link](https://disk.yandex.ru/d/dV5YkIKnPkysDQ?w=1)
### Inference and evaluation
```bash 
cd repos/biaffine-ner
python evaluate.py experiment5
python inference.py experiment5 data/10.04.2021/train.jsonlines logs/experiment5/exp5-train-inference.jsonlines
```

## Pyramid NER
[link to article](https://www.notion.so/75aa5b16cf7b4e4d9d687d28dd63ca34#c884ddf9473b4a05a5d49dd448078e92)

[link to original implementation (pytorch)](https://github.com/LorrinWWW/Pyramid)
### Training
[train-dummy.sh](repos/Pyramid/train-dummy.sh) - training with fasttext embeddings only
[train.sh](repos/Pyramid/train.sh) - training with ruBERT and fasttext embeddings
if u want to use bert, then precompute bert embeddings with 
### Pretrained models
* char-cnn + fasttext embeddings (cc300ru) - [link](https://disk.yandex.ru/d/ViLxIGswG6ThIQ?w=1)
* char-cnn + fasttext (cc300ru) + ruBERT embeddings - [link](https://disk.yandex.ru/d/Z4n4R3UwAxNHfw?w=1)
### Inference and evaluation
how to infer

## Seq2Seq
aka Neural Architectures for Nested NER through Linearization
[link to article](https://www.aclweb.org/anthology/P19-1527.pdf)  
[link to original implementation (tensorflow<2.0)](https://github.com/ufal/acl2019_nested_ner)
### Training

[train-dummy.sh](repos/acl2019_nested_ner/train-dummy.sh) - training with fasttext embeddings only
<!-- [train.sh](repos/acl2019_nested_ner/train.sh) - training with ruBERT and fasttext embeddings -->

### Pretrained models
* char-cnn + fasttext embeddings (cc300ru) - [link](https://disk.yandex.ru/d/HcKhbD0HsAHtuA?w=1)
* char-cnn + fasttext (cc300ru) + ruBERT embeddings - TBA
### Inference and evaluation
how to infer
# Results

Add files with results per category
link to files with per cat results

From 10.04.2021 on test set

| MODEL                            | PREC  |  REC  |  F1   | SCORES                                          | CHECKPOINT                                          |
| -------------------------------- | :---: | :---: | :---: | ----------------------------------------------- | --------------------------------------------------- |
| biaffine-ner + fasttext          | 78.8  | 71.8  | 75.13 | [link](https://disk.yandex.ru/d/PLhGycbc8bggHQ) | [link](https://disk.yandex.ru/d/Q4O2MVvWppvoUA?w=1) |
| biaffine-ner + fasttext + ruBERT | 81.92 | 71.54 | 76.38 | [link](https://disk.yandex.ru/d/3O9z0TsmtsbBhg) | [link](https://disk.yandex.ru/d/dV5YkIKnPkysDQ?w=1) |
| pyramid-ner + fasttext           | 72.70 | 63.01 | 67.51 | [link](https://disk.yandex.ru/d/V-IdxzmxXgxsPg) | [link](https://disk.yandex.ru/d/ViLxIGswG6ThIQ?w=1) |
| pyramid-ner + fasttext + ruBERT  | 77.73 | 70.97 | 74.19 | [link](https://disk.yandex.ru/d/woZeQA0laUNFVA) | [link](https://disk.yandex.ru/d/Z4n4R3UwAxNHfw?w=1) |
| seq2seq-ner + fasttext           | 74.01 | 71.51 | 72.74 | TBA                                             | [link](https://disk.yandex.ru/d/HcKhbD0HsAHtuA?w=1) |
| seq2seq-ner + fasttext + ruBERT  |  TBA  |  TBA  |  TBA  | TBA                                             | TBA                                                 |


<!-- Inference examples
Medical dataset (link) -->