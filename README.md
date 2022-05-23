# Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations

Official Code Repository for the paper "Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations": https://arxiv.org/abs/2202.02514 (To appear at ICML 2022).

In this repository, we implement the *Graph Diffusion via the System of SDEs* (GDSS).

## Abstract

Generating graph-structured data requires learning the underlying distribution of graphs. Yet, this is a challenging problem, and the previous graph generative methods either fail to capture the permutation-invariance property of graphs or cannot sufficiently model the complex dependency between nodes and edges, which is crucial for generating real-world graphs such as molecules. To overcome such limitations, we propose a novel score-based generative model for graphs with a continuous-time framework. Specifically, we propose a new graph diffusion process that models the joint distribution of the nodes and edges through a system of stochastic differential equations (SDEs). Then, we derive novel score matching objectives tailored for the proposed diffusion process to estimate the gradient of the joint log-density with respect to each component, and introduce a new solver for the system of SDEs to efficiently sample from the reverse diffusion process. We validate our graph generation method on diverse datasets, on which it either achieves significantly superior or competitive performance to the baselines. Further analysis shows that our method is able to generate molecules that lie close to the training distribution yet do not violate the chemical valency rule, demonstrating the effectiveness of the system of SDEs in modeling the node-edge relationships.


### Contribution

+ We propose a novel score-based generative model for graphs that overcomes the limitation of previous generative methods, by introducing a diffusion process for graphs that can generate node features and adjacency simultaneously via the system of SDEs.
+ We derive novel training objectives to estimate the gradient of the joint log-density for the proposed diffusion process and further introduce an efficient integrator to solve the proposed system of SDEs.
+ We validate our method on both synthetic and real-world graph generation tasks, on which ours outperforms existing graph generative models.

## Dependencies

GDSS is built in **Python 3.7.0** and **Pytorch 1.8.0**. Please use the following command to install the requirements:

```python
pip install -r requirements.txt
```


## Running Experiments


### Preparations

To generate the datasets  for training models, please run the following line:

```sh
sh ./scirpts/data.sh dataset_name
```

We provide four generic graph datasets: Ego-small, Community_small, ENZYMES, and Grid.
To reproduce our results, please use the provided data.

To compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html) for the evaluation, please run the following lines:

```python
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```


### Training 

We provide the commands for the following tasks: Generic Graph Generation and Molecule Generation.

For each command, the first argument denotes the name of the dataset, second argument denotes the gpu id, and the third argument denotes the experiment seed.

Generic Graph Generation
```sh
sh ./scripts/train.sh community_small 0 42
```

Molecule Generation
```sh
sh ./scripts/train.sh community_small 0 42
```

### Generation and Evaluation 

For each command, the first argument denotes the gpu id, and the second argument denotes the experiment seed.

```sh
sh ./scripts/sample.sh 0 42
```

### Configurations

The configurations are in the `config/` directory in the `YAML` format. 
Hyperparameters used in the experiments are specified in the Appendix C. in our paper.

## Pretrained checkpoints

We additionally provide checkpoints of the pretrained models on ZINC250k dataset [here](https://drive.google.com/drive/folders/1gSM66ZZVfyUcFYkSAKmWl97M4YY8mKUu?usp=sharing).

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@article{jo2022GDSS,
  author    = {Jaehyeong Jo and
               Seul Lee and
               Sung Ju Hwang},
  title     = {Score-based Generative Modeling of Graphs via the System of Stochastic
               Differential Equations},
  journal   = {arXiv:2202.02514},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.02514}
}
```