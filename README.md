# dp-llm


This repository contains an implementation of __Interpretable and Differentially Private Predictions__. The paper by [Frederik Harder](https://ei.is.tuebingen.mpg.de/person/fharder), [Matthias Bauer](https://ei.is.tuebingen.mpg.de/person/bauer) and [Mijung Park](https://ei.is.mpg.de/~mpark) was published at AAAI 2020 and is available at https://aaai.org/ojs/index.php/AAAI/article/view/5827 


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

`src/model_eval/` contains scripts used for generating the Mnist graphs and visualizations shown in the paper
- `filter_visualization.py` generates all filter visualizations for LLMs (Fig. 3-5)
- `gradient_attribution.py` generates SmoothGrad and Integrated Gradient attributions for a reference CNN or LLMs (Fig.4)
- `params_vs_accuracy.py` trains multiple LLMs under changing random seeds and aggregates test accuracies (Fig. 2)

`src/moments_accountant/` contains the code used for computing the privacy guarantee epsilon values, given a specific delta and sigma. 
- `moments_accountant.py` is taken from repository `https://github.com/tensorflow/models/tree/master/research/differential_privacy` which was uploaded by Abadi et al. and which has since been removed
- `ma_main.py` provides an interface to call the moments calculation functions


If you have any questions or comments, please don't hesitate to contact [Frederik Harder](https://ei.is.tuebingen.mpg.de/person/fharder).
