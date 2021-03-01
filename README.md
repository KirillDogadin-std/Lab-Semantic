## Purpose of the code

The purpose consists of the following goals:
- Convert the original implementation of the _MultiKE_ method from the paper
written in `tensorflow 1.x`: reimplement the approach in `pytorch`.

- Modify the initial approach of using the _TransE_ model with the _MDE_ model.

### Bonus work

- Add supports for new benchmark datasets (_OpenEA_) as well as smaller version of original datasets to both implementations.

- Add the _MDE_ model to the original implementation.

### Links to the articles: 

- MultiKE: https://www.ijcai.org/Proceedings/2019/0754.pdf
  - Git repository: https://github.com/nju-websoft/MultiKE

- MDE: https://ecai2020.eu/papers/1271_paper.pdf
  - Git repository: https://github.com/mlwin-de/MDE_adv

## How to run the code?

### First and foremost

To save your time and effort before trying to run the model on your local machine, please use the following link:

https://colab.research.google.com/drive/1E0rUGU6rGfOG5vMkug3u4vwxfIKytJRm?usp=sharing

This is a notebook in Google Colab that requires you to just hit 'Run', without any need of setting up the environment on your local machine.

If this option fails, please send an email asap, so that the issue with the Colab can be resolved.
In the meantime, proceed to the below instructions on how to run the code.

## Expected results

The expected result of the code execution generally is:
  - Training process for entity alignment
  - The evaluation step of entity matching provides metrics: `Hits@`, mean rank (`MR`), mean reciprocal rank (`MRR`)
  - `pytorch` implementation reproduces the result from the original implementation
  - `MDE` has worse performance compared with `TransE`

Results of the implementations on D-W 15K dataset:

|implementation|Hits@1|Hits@5|Hits@10|Hits@50|MR|MRR|
|:---: |:---: |:---: | :---: | :---: | :---: | :---: |
|ours|90.7|95.0|95.9|97.6|18.8|92.7|
|ours-MDE|83.4|86.6|87.6|90.3|170.1|85.0|
|original|90.4|94.0|95.0|97.0|20.3|92.1|


### Requirements

* Python 3
* TensorFlow>=1.8 / PyTorch>=1.6
* Numpy
* Scikit-learn
* Levenshtein
* Gensim

Setting up the environment to run the code:

- Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
- Create an environment for python3.7
  - run `conda create -n multike python=3.7`
  - choose the environment `conda activate multike`
- Install tensorflow 1.x : `conda install tensorflow-gpu=1.15`
- Install pytorch:
  - Go to the link :https://pytorch.org/get-started/locally/
  - Set up the parameters interactively. E.g., for Linux it is `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

- Other requirements
    ```bash
    pip install -r requirements.txt
    ```

Download the following archive, extract and place them in the repository *data/* directory - you will need it later.

- pre-trained word vectors, filename to download: `wiki-news-300d-1M.vec.zip`
  - Link: https://fasttext.cc/docs/en/english-vectors.html
- OpenEA datasets
  - Link: https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0


You are set up.

### Running

**Important instructions about the set up:**

In the case you placed the required files mentioned above correctly, you do not need to edit the configuration file `args.json`. 
For the pytorch implementation the file is located in `code/pytorch/` directory and for tensorflow in `code/` directory. 
Please, open it and edit accordingly:

- The file location paths
```json
  "dataset": "path/to/training/data",
  "output": "path/to/output/results/",
  "word2vec": "path/to/wiki-news-300d-1M.vec",
  "dataset_division": "631/",
```
  - `dataset_division` path is relative to the `dataset` path
  - by default, training data is contained in `D_Y_15K_V1`. This lies in `data/` directory of the repository.
  - You should have downloaded `wiki-news-300d-1M.vec` by now. Please specify the path to it in `word2vec`.
  - The original dataset is very large and most computers cannot handle it due to insufficient RAM resource. To run with smaller dataset, refer to the
  sections of this readme with below titles:
    - `Additional related information`, subsection `Dataset`
    - `Running with smaller dataset`

**General instructions:**

To run the experiments in PyTorch, use:

```
python code/main.py --data dataset_path --method method --mode mode
```
For TensorFlow, use:
```
python code/run.py --data dataset_path --method method --mode mode
```

* dataset_path: the path of dataset to run;
* method: training method, using either ITC or SSL;
* mode: embedding mode, using either TransE or MDE.

For example, to run the experiments on D-Y-15K with ITC method and TransE mode, use:

```
python code/main.py --data data/D_Y_15K_V1/ --method ITC --mode TransE
```

You can redirect output to file with `> log.txt`. When run is finished, see the log in the created `./log.txt` file.

**Running with smaller dataset**

- The repository contains the directory `mock_data/` - it can be specified as `dataset_path`.
- Please adjust the following fields of learning rates in `args.json` (due to the significantly smaller dataset):
  ```
    "learning_rate": 0.1,
    "relation_learning_rate": 0.1,
    "ITC_learning_rate": 0.1
  ```
- Then run the code as described above.
- Note that it's possible to generate other 'small' datasets with scripts provided in the same directory. You can address the README in this directory.
  - If the RAM is still not sufficient, you need to regenerate the dataset (perhaps even a couple of times, as it runs with randomization).
  - For different datasets use different learning rates in `args.json`.

### Scenarios of runs
#### Ideal scenario

You have sufficient RAM resource, you run both pytorch and tensorflow versions and see that pytorch behaves
comparably - losses do not differ much, matched entities/relations/attributes do not differ much.

#### Smaller dataset

You can see the decreasing loss in the log.
Running tensorflow version also provides the same loss behaviour.

Depending on the `args.json`, matching results can differ.

### Log file contains:

#### Literal encoder training
```
epoch 1 of literal encoder, loss: 405542.1463, time: 2.7521s
epoch 2 of literal encoder, loss: 326954.1061, time: 2.4722s
```
#### MultiKE training:
```
epoch 1:
epoch 1 of rv, avg. loss: 15361.9040, time: 2.6948s
epoch 1 of ckgrtv, avg. loss: 16024.1237, time: 0.2528s
epoch 1 of av, avg. loss: 1310.5740, time: 3.1883s
epoch 1 of ckgatv, avg. loss: 12874.6576, time: 0.5854s
epoch 1 of cnv, avg. loss: 23811.5635, time: 0.4768s
epoch 2:
epoch 2 of rv, avg. loss: 12546.4685, time: 2.8604s
epoch 2 of ckgrtv, avg. loss: 11283.0437, time: 0.2521s
epoch 2 of av, avg. loss: 1310.4260, time: 3.0020s
epoch 2 of ckgatv, avg. loss: 12863.3533, time: 0.5715s
epoch 2 of cnv, avg. loss: 14076.8319, time: 0.4758s
```
Loss should decrease over time.

## Additional related information

### MultiKE
Source code and datasets for IJCAI-2019 paper "_[Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/0754.pdf)_".

### Dataset
We used two datasets, namely DBP-WD and DBP-YG, which are based on DWY100K proposed in [BootEA](https://www.ijcai.org/proceedings/2018/0611.pdf). 

#### DBP-WD and DBP-YG
In "data/BootEA_datasets.zip", we provide the full data of the two datasets that we used. Each dataset has the following files:

* ent_links: all the entity links without training/test/valid splits;
* 631: entity links with training/test/valid splits, contains three files, namely train_links, test_links and valid_links;
* attr_triples_1: attribute triples in the source KG;
* attr_triples_2: attribute triples in the target KG;
* entity_local_name_1: entity local names in the source KG, list of pairs like (entity \t local_name);
* entity_local_name_2: entity local names in the target KG;
* predicate_local_name_1: predicate local names in the source KG, list of pairs like (predicate \t local_name);
* predicate_local_name_2: predicate local names in the target KG.
* rel_triples_1: relation triples in the source KG, list of triples like (h \t r \t t);
* rel_triples_2: relation triples in the target KG;

The raw datasets of DWY100K can also be found [here](https://github.com/nju-websoft/BootEA/tree/master/dataset).

#### OpenEA
Datasets proposed in [OpenEA](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf), the datasets consist of 4 datasets can be downloaded from [Dropbox](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0).
Each dataset has the following files:

* ent_links: entity alignment between KG1 and KG2
* 721_5fold: entity alignment with test/train/valid (7:2:1) splits
* attr_triples_1: attribute triples in KG1
* attr_triples_2: attribute triples in KG2
* rel_triples_1: relation triples in KG1
* rel_triples_2: relation triples in KG2

Dataset overview:

*#* Entities | Languages | Dataset names
:---: | :---: | :---: 
15K | Cross-lingual | EN-FR-15K, EN-DE-15K
15K | English | D-W-15K, D-Y-15K
100K | Cross-lingual | EN-FR-100K, EN-DE-100K
100K | English-lingual | D-W-100K, D-Y-100K

More information about datasets can be found [here](https://github.com/nju-websoft/OpenEA).

#### Package description

```
code/
├── pytorch/: package of the implementations for datasets, model, loss, literal encoder, training, and predicate alignment
│   ├── finding/: package of the implementations for searching alignment and similarity between the two collections of embeddings
│   ├── load/: package of the implementations for read and load kgs
```

## In case of problems or questions

contact s6kidoga@uni-bonn.de

## Citation

```
@inproceedings{MultiKE,
  author    = {Qingheng Zhang and Zequn Sun and Wei Hu and Muhao Chen and Lingbing Guo and Yuzhong Qu},
  title     = {Multi-view Knowledge Graph Embedding for Entity Alignment},
  booktitle = {IJCAI},
  pages     = {5429--5435},
  year      = {2019}
}
@inproceedings{sadeghi2020mde,
  title     = {MDE: Multiple Distance Embeddings for Link Prediction in Knowledge Graphs},
  author    = {Sadeghi, Afshin and Graux, Damien and Shariat Yazdi, Hamed and Lehmann, Jens},
  booktitle = {24th European Conference on Artificial Intelligence, ECAI},
  year      = {2020},
  url       = {http://ecai2020.eu/papers/1271_paper.pdf}
}
@article{OpenEA,
  author    = {Zequn Sun and Qingheng Zhang and Wei Hu and Chengming Wang and Muhao Chen and Farahnaz Akrami and Chengkai Li},
  title     = {A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs},
  journal   = {Proceedings of the VLDB Endowment},
  volume    = {13},
  number    = {11},
  pages     = {2326--2340},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2326-sun.pdf}
}
```
