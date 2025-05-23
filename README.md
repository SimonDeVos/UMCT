# Uplift modeling with continuous treatments: A predict-then-optimize approach</br><sub><sub>Simon De Vos, Christopher Rickermann, Stefan Lessmann, Wouter Verbeke </sub></sub>  

This paper introduces a predict-then-optimize framework for uplift modeling with continuous treatments, combining causal machine learning to estimate conditional average dose responses (CADRs) and integer linear programming to optimize dose allocation, enabling effective and adaptable decision-making across diverse applications while considering constraints like fairness and instance-dependent costs.

A preprint is available on [ArXiv](https://arxiv.org/abs/2412.09232).


## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies (can take some minutes).

The code uses the Gurobi optimization software (version 11.0.1). Gurobi offers free [academic licenses](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Instructions:
- In [main.ipynb](main.ipynb):
  - Set the project directory to your custom folder. E.g., `DIR = r'C:\Users\...\...\...'`
  - Specify model in cell three. E.g.: `model_settings = {"model_type": SLearner, "model_name": "randomf"}`
  - Set data, methods, optimization, plot, and project configurations in their corresponding [.yaml files](https://github.com/SimonDeVos/UMCT/tree/master/config)

## Repository Structure
This repository is organized as follows:
```bash
|- config/
    |- data/config.yaml          # dataset configurations
    |- methods/config.yaml       # method configurations
    |- optimization/config.yaml  # optimization configurations
    |- plot/config.yaml          # plot configurations
    |- project/config.yaml       # project configurations
|- data/
    |- ihdp_s_1/             
        |- IHDP-S-1.csv          # IHDP dataset
|- experimental results/         # figures as displayed in the paper, organized per experiment
    |- experiment 0/             # figure of Appendix C
    |- experiment 1/     
    |- experiment 2/
    |- experiment 3/
|- main.ipynb                    # Main notebook -- entry point for running experiments
|- src/
    |- data/
        |- utils/
            |- __init__.py
            |- _dclasses.py
            |- _functional.py
            |- _sampling.py
        |- ihdp_1.py
        |- utils.py
    |- methods/
        |- utils/
            |- classes.py
            |- drnet.py
            |- losses.py
            |- regressors.py
            |- vcnet.py
        |- neural.py
        |- other.py
    |- metrics/
        |- metrics.py
    |- optimization/
        |- optimizer.py
        |- utils.py
    |- utils/
        |- metrics.py
        |- plotting.py
        |- setup.py
        |- training.py
        |- viz.py
```
