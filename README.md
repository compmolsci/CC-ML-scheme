# CC-ML-scheme
Machine Learning, Synergetics and Non-Linear Dynamics based Accelerated Coupled Cluster Theory


The iteration scheme associated with single reference coupled cluster theory has been analysed using nonlinear dynamics. The phase space analysis indicates the presence of a few
significant cluster amplitudes, mostly involving valence excitations, that dictate the dynamics, while all other amplitudes are enslaved. Starting with a few initial iterations to establish the inter-relationship among the cluster amplitudes, a supervised Machine Learning scheme with polynomial Kernel Ridge Regression model has been employed to express each of the enslaved amplitudes uniquely in terms of the former set of amplitudes. The subsequent coupled cluster iterations are restricted solely to determine those significant excitations, and the enslaved amplitudes are determined through the already established functional mapping. We will show that our hybrid scheme leads to significant reduction in computational time without sacrificing the accuracy. 

Running the code with CC-ML:
1. Setup molecule according to `pyscf` in `inp.py` file
2. Input KRR model parameters in `inp.py` and number of training and discarded iterations
3. `python3 1main.py` for running the CC-ML iteration

Running the code for Exact CCSD:
Skip step 2, and then `python3 main.py`
