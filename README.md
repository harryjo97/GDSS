# Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations

Official Code Repository for the paper "Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations": https://arxiv.org/abs/2202.02514 (ICML 2022).

In this repository, we implement the *Graph Diffusion via the System of SDEs* (GDSS).

## Abstract

Generating graph-structured data requires learning the underlying distribution of graphs. Yet, this is a challenging problem, and the previous graph generative methods either fail to capture the permutation-invariance property of graphs or cannot sufficiently model the complex dependency between nodes and edges, which is crucial for generating real-world graphs such as molecules. To overcome such limitations, we propose a novel score-based generative model for graphs with a continuous-time framework. Specifically, we propose a new graph diffusion process that models the joint distribution of the nodes and edges through a system of stochastic differential equations (SDEs). Then, we derive novel score matching objectives tailored for the proposed diffusion process to estimate the gradient of the joint log-density with respect to each component, and introduce a new solver for the system of SDEs to efficiently sample from the reverse diffusion process. We validate our graph generation method on diverse datasets, on which it either achieves significantly superior or competitive performance to the baselines. Further analysis shows that our method is able to generate molecules that lie close to the training distribution yet do not violate the chemical valency rule, demonstrating the effectiveness of the system of SDEs in modeling the node-edge relationships.


### Contribution

+ We propose a novel score-based generative model for graphs that overcomes the limitation of previous generative methods, by introducing a diffusion process for graphs that can generate node features and adjacency simultaneously via the system of SDEs.
+ We derive novel training objectives to estimate the gradient of the joint log-density for the proposed diffusion process and further introduce an efficient integrator to solve the proposed system of SDEs.
+ We validate our method on both synthetic and real-world graph generation tasks, on which ours outperforms existing graph generative models.

## Dependencies

GDSS is built in **Python 3.7.0** and **Pytorch 1.10.1**. Please use the following command to install the requirements:

```sh
pip install -r requirements.txt
```

For molecule generation, additionally run the following command:

```sh
conda install -c conda-forge rdkit=2020.09.1.0
pip install git+https://github.com/fabriziocosta/EDeN.git --user
```


## Running Experiments


### Preparations

To generate the generic graph datasets for training models, run the following command:

```sh
python data/data_generators.py --dataset ${dataset_name}
```

We provide four generic graph datasets: Ego-small, Community_small, ENZYMES, and Grid.
To reproduce our results, please use the provided data.

To compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html) for the evaluation, run the following command:

```sh
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

To preprocess the molecular graph datasets for training models, run the following command:

```sh
python data/preprocess.py --dataset ${dataset_name}
python data/preprocess_for_nspdk.py --dataset ${dataset_name}
```


### Configurations

The configurations are provided on the `config/` directory in `YAML` format. 
Hyperparameters used in the experiments are specified in the Appendix C of our paper.


### Training

We provide the commands for the following tasks: Generic Graph Generation and Molecule Generation.

```sh
sh scripts/train.sh ${dataset_name} ${gpu_id} ${seed}
```

<!-- Note that training score-based models on ZINC250k dataset requires gpu memory larger than GB
We provide data parallel training code for training. -->

### Generation and Evaluation

To sample graphs using the trained score models, first modify `config/sample.yaml` accordingly, then run the following command.

```sh
sh scripts/sample.sh ${gpu_id}
```

## Pretrained checkpoints

We provide checkpoints of the pretrained models on the `checkpoints/` directory, which are used in the main experiments.

+ `ego_small/gdss_ego_small.pth`
+ `community_small/gdss_community_small.pth`
+ `ENZYMES/gdss_enzymes.pth`
+ `grid/gdss_grid.pth`
+ `ZINC250k/gdss_zinc250k.pth` 

We also provide checkpoints of improved GDSS that uses GMH blocks instead of GCN blocks in $s_{\theta,t}$ (i.e., that uses `ScoreNetworkX_GMH` instead of `ScoreNetworkX`). The numbers of training epochs are 800 and 1000 for $s_{\theta,t}$ and $s_{\phi,t}$, respectively. For this checkpoint, use Rev. + Langevin solver and set `snr` as 0.2 and `scale_eps` as 0.8.

+ `gdss_zinc250k_v2.pth` 

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
