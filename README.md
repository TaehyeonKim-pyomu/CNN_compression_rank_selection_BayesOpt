# BayesOpt-based Global Optimal Rank Selection for Compression of CNNs
T. Kim, J. Lee and Y. Choe, "Bayesian Optimization-Based Global Optimal Rank Selection for Compression of Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 17605-17618, 2020, doi: 10.1109/ACCESS.2020.2968357.

# Abstract 
Finding the optimal rank is a crucial problem beacause the rank is the only hyperparameter for controlling computational complexity and accuracy in compressed CNNs. To solve this problem, we propose a global optimal rank selection method based on Bayesian optimization. By utilizing both a simple objective function and a proper optimization scheme, the proposed method produces a global optimal rank that provides a good trade-off between computational complexity and accuracy degradation. 

# Usage
- Decompose a pretrained vgg16 model:  ``python main.py`` 
- torch: 1.0.0 version 
- tensorly: 0.4.5 version 
- GPyOpt: 1.2.5 version 

# Reference 
- Global optimal rank selection via BayesOpt: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8964358
- CNN compression via Tucker decomposition: https://arxiv.org/abs/1511.06530
- Pytorch-tensor-decomposition: https://github.com/jacobgil/pytorch-tensor-decompositions
- Tensorly: https://github.com/tensorly/tensorly
- GPyOpt: https://github.com/SheffieldML/GPyOpt

