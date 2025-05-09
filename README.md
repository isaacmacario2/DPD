# Introduction
This repository is intended for the implementation of a Digital Pre-Distortion (DPD) for compensation of non-linearities, inserted by a power amplifier (PA), with several types of adaptive filtering algorithms.

The DPDs of this repository follow the Indirect Learning Architecture (ILA) model, as shown in the figure below.

![image](https://github.com/user-attachments/assets/88d4966d-faad-475e-911e-1c4eeb7ce5e8)

In this repository, two codes for pre-distortion were made, the first is simulation.m, which uses a database of the NXP Airfast LDMOS Doherty PA power amplifier provided by Math Works at https://www.mathworks.com/help/comm/ug/power-amplifier-characterization.html. The second code, DPD_RFWeblab.m, is made with real data from the Cree CGH40006-TB amplifier, remotely, through the website https://dpdcompetition.com/rfweblab/.

To carry out the second code, the following structure is used:

![image](https://github.com/user-attachments/assets/35d0c5f2-c3fd-4bbb-9d82-c180e5e3bb33)

# EX-QKRLS



# How to use

Download this repository and leave all the codes in the same folder. For the DPD_RFWeblab.m code, an internet connection is required and the system status provided by the website https://dpdcompetition.com/rfweblab/ must be up and running.

# References

D. R. Morgan, Z. Ma, J. Kim, M. G. Zierdt and J. Pastalan, "A Generalized Memory Polynomial Model for Digital Predistortion of RF Power Amplifiers", in IEEE Transactions on Signal Processing, vol. 54, no. 10, pp. 3852-3860, Oct. 2006.

P. S. R. Diniz, "Adaptive Filtering: Algorithms and practical implementation}. 4. ed. New York: Springer, 2013.

J. A. Apolinário Jr., "QRD-RLS Adaptive Filtering". New York: Springer, 2009.

P. S. R. Diniz, M. L. R. Campos,  W. A. Martins, M. V. S. Lima and J. A. Apolinário Jr., "Online Learning and Adaptive Filters". New York: Wiley, 2023.

W.Liu, J. C. Príncipe and S. Haykin, "Kernel Adaptive Filtering: A comprehensive introduction". 1. ed. New Jersey: Wiley, 2010.

A. Sayed, \textit{Adaptive Filters}. 1. ed. New Jersey: Wiley, 2008.

J. Platt, "A resource-allocating network for function interpolation". in Neural Computation - NECO, v. 3, p. 213–225, 06 1991.

B. Chen, S. Zhao, P. Zhu and J. C. Príncipe, "Quantized Kernel Recursive Least Squares Algorithm", in IEEE Transactions on Neural Networks and Learning Systems, vol. 24, no. 9, pp. 1484-1491, Sept. 2013.

C. Tarver, "GMP DPD Using Indirect Learning Architecture Matlab Library," sept. de 2019. Avalable in: <https://github.com/ctarver/ILA-DPD>. 

# How to Cite
@misc{MacarioDPD,
  author       = {I. M. S. Gouveia, J. A. Apolinário Jr., C. A. B. Saunders Filho},
  title        = {Pré-Distorção Digital Baseada em Kernel: Uma Abordagem com o Algoritmo EX-QKRLS},
  month        = may,
  year         = 2025,
}
