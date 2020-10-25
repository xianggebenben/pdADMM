# pdADMM: parallel deep learning Alternating Direction Method of Multipliers

This is a implementation of ADMM to achieve  model parallelism for the deep fully-connected neural network, as described in our paper:

Junxiang Wang, Zheng Chai, Yue Cheng, and Liang Zhao. Toward Model Parallelism for Deep Neural Network based on Gradient-free ADMM Framework. (ICDM 2020)

serial_pdADMM  is the source code of serial implementation of the pdADMM algorithm.

parallel_pdADMM is the source code of parallel implementation of the pdADMM algorithm.

## Cite
Please cite our paper if you use this code in your own work:

@inproceedings{wang2020toward,

author = { Wang, Junxiang and Chai, Zheng and Cheng, Yue and Zhao, Liang},

title = {Toward Model Parallelism for Deep Neural Network based on Gradient-free ADMM Framework},

year = {2020},

booktitle = {Proceedings of the 20th IEEE International Conference on Data Mining},

location = {Sorrento, Italy},

series = {ICDM ’20}

}
