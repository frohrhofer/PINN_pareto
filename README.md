# On the Apparent Pareto Front of Physics-Informed Neural Networks
Code to Paper

## Abstract
Physics-informed neural networks (PINNs) have emerged as a promising deep learning method, capable of solving problems which are governed by differential equations.
Despite their recent advance, it is widely acknowledged that PINNs are difficult to train and often require a careful tuning of loss weights when data and physics loss functions are combined by scalarization of a multi-objective (MO) problem.
In this paper, we aim at understanding how parameters of the physical system, such as characteristic length and time scales, the computational domain, and coefficients of differential equations, affect the MO optimization and optimal choice of loss weights.
Through a theoretical examination of where these system parameters appear in the PINN training, we find that they effectively and individually scale the loss residuals, causing imbalances in the MO optimization for certain choices of system parameters.
The immediate effects of this are reflected in the apparent Pareto front, which we define as the set of loss values achievable with gradient-based training and visualize accordingly.
We empirically verify that loss weights can be used successfully to compensate for the scaling of system parameters, and enable the selection of an optimal solution on the apparent Pareto front that aligns well with the physically valid solution.
We further demonstrate that by altering the system parameterization, the apparent Pareto front can shift and exhibit locally convex parts, resulting in a wider range of loss weights for which gradient-based training becomes successful. 
This work explains the effects of system parameters on the MO optimization in PINNs, and highlights the utility of proposed loss weighting schemes.


## Requirements
- tensorflow>2.2
- numpy
- scipy
- pyDOE
- pyyaml
