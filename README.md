# PIE: a Parameter and Inference Efficient Solution for Large Scale Knowledge Graph Embedding Reasoning
Paper link: https://arxiv.org/abs/2204.13957

## Results
The results of PIE powered TransE model is as follows.

### ogb-lsc-wikikg90mv2
| | #Parameters | Test MRR | Valid MRR |
|:------:|:------:|:------:|:------:|
| TransE-shallow-PIE | 18,247,074,200 | 0.1883 | 0.2342 |

#### System requirements:
It's recommended to use a machine with over 400G memory to reproduce the results.

## Reproducing the Results on WikiKG90Mv2
---
#### 1, Entity Typing
The model implementation is based on [**PathCon**](https://github.com/hyren/PathCon). Thanks for their contributions.

```bash
cd entity_typing && sh run.sh DATA_PATH
```

The ${DTA_PATH} should conatin train_hrt.npy.

After model training and inference, the code will output the distributuon p(r|e) for each entity. As the dataset is too large, we save the results with 15 sparse matrices. Please use src/cat_npz.py to concatenate all the sparse matrices.
`cd src && python/cat_npz.py $SAVE_PATH`
This will save the $p(r|e)$ matrix for all entities  at ${SAVE_PATH}/e2r_scores.npz.

#### 2, Candidate Generation
The sampling implementation is based on [**dgl-ke**](https://github.com/awslabs/dgl-ke). Thanks for their contributions.

```bash
cd candidate && sh run.sh DATA_PATH
```

The ${DATA_PATH} should contain folders, such as 'raw' and 'processed'.
As we utilize multi processer to sampling the candidates and each processor saves the result individually, please use script src/cat.py to concatenate all the candidates to one file.

#### 3, Knowledge Graph Embedding Reansoning
The KGE implementation is based on the official baseline code of WikiKG90M [**OGB**](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/wikikg90m). Thanks for their contributions.


```bash 
cd wikikg90m-v2
sh install_dgl.sh
sh run.sh DATA_PATH SAVE_PATH
sh run_test.sh SAVE_PATH DATA_PATH VAL_CANDIDATE_PATH TEST_CANDIDATE_PATH
```
