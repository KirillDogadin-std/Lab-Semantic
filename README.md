# MultiKE
Source code and datasets for IJCAI-2019 paper "_[Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/0754.pdf)_".

## Dataset
We used two datasets, namely DBP-WD and DBP-YG, which are based on DWY100K proposed in [BootEA](https://www.ijcai.org/proceedings/2018/0611.pdf). 

### DBP-WD and DBP-YG
In "data/BootEA_datasets.zip", we give the full data of the two datasets that we used. Each dataset has the following files:

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

### OpenEA
Datasets proposed in [OpenEA](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf), the datasets can be downloaded from [Dropbox](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0).
Each dataset has the following files:

* ent_links: entity alignment between KG1 and KG2
* 721_5fold: entity alignment with test/train/valid (7:2:1) splits
* attr_triples_1: attribute triples in KG1
* attr_triples_2: attribute triples in KG2
* rel_triples_1: relation triples in KG1
* rel_triples_2: relation triples in KG2

More information about datasets can be found [here](https://github.com/nju-websoft/OpenEA).

## Dependencies
* Python 3
* TensorFlow 1.x / PyTorch 1.x
* Numpy
* Scikit-learn
* Levenshtein
* Gensim

## Run

To run the experiments, use:

    python code/run.py --data dataset_path --method method --mode mode
* dataset_path: the path of dataset to run;
* method: training method, using either ITC or SSL;
* mode: embedding mode, using either TransE or MDE.

For example, to run the experiments on DBP-WD with ITC method and TransE mode, use:

    python code/run.py --data data/BootEA_DBP_WD_100K/ --method ITC --mode TransE

## Citation
If you use this model or code, please kindly cite it as follows:      

```
@inproceedings{MultiKE,
  author    = {Qingheng Zhang and Zequn Sun and Wei Hu and Muhao Chen and Lingbing Guo and Yuzhong Qu},
  title     = {Multi-view Knowledge Graph Embedding for Entity Alignment},
  booktitle = {IJCAI},
  pages     = {5429--5435},
  year      = {2019}
}
```
