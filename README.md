# dp-llm


This repository contains an implementation of __Interpretable and Differentially Private Predictions__. The paper is available at https://arxiv.org/abs/1906.02004 

## Setup

- The LLM Mnist model is ready to use.
- user friendly scripts for reproduction of experimental results on Mnist and Fashion-Mnist coming soon. 

### Dependencies
    python 3.6
    torch 1.0.1.post2
    numpy 1.14.0
    scipy 1.1.0
    matplotlib 2.2.2 (plotting only)
    mpmath 1.0.0 (moments calculation only)


## Repository Structure

`src/` contains all code
- `llm_mnist_model.py` contains the Locally Linear Maps (LLM) model used for Mnist experiments
- `llm_mnist_main.py` can be called to train LLM models on Mnist and Fashion-Mnist
- `mnist_cnn.py` contains and trains the reference CNN model used in the paper
- `utils.py` contains various utility functions.
`src/model_eval/` contains scripts used for generating the graphs and visualizations shown in the paper
- `filter_visualization.py`
- `gradient_attribution.py`
- `params_vs_accuracy.py`
`src/moments_accountant/` contains the code used for computing the privacy guarantee epsilon values, given a specific delta and sigma. 
- `moments_accountant.py` is taken from repository `https://github.com/tensorflow/models/tree/master/research/differential_privacy` which was uploaded by Abadi et al. and which has since been removed
- `ma_main.py` provides an interface to call the moments calculation functions



If you have any questions or comments, please don't hesitate to contact [Frederik Harder](https://ei.is.tuebingen.mpg.de/employees/fharder).
