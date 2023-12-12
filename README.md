# CSC591-Efficient Tensor Processing for AI: Final project
Efficient Tensor Processing for AI Final Project Source code:

Please find different components of the code in these repositories.
The repositories' contents are organized & explained as below.

## Tensorized transformer code:

Find the main repository at: https://github.com/srina1h/Tensor_Contract_Optim
Work is done on separate branches. Main is kept as-is from the main repository that its forked from (https://github.com/ziyangjoy/Tensor_Contract_Optim)

### Branches:
- Branch ct_con: https://github.com/srina1h/Tensor_Contract_Optim/tree/ct_con - Tensor contraction using cupy cutensor on a single tensordot operation
- Branch ct_con2: https://github.com/srina1h/Tensor_Contract_Optim/tree/ct_con2 - Tensor contraction using cupy cutensor on another tensordot operation
- Branch dim_test: https://github.com/srina1h/Tensor_Contract_Optim/tree/dim_test - To understand dimensions of different contractions
- Branch profiling: https://github.com/srina1h/Tensor_Contract_Optim/tree/profiling - Profiling using PyTorch profiler
- Branch nvtx_profile: https://github.com/srina1h/Tensor_Contract_Optim/tree/nvtxprofile - Profiling using Nsight systems
- Branches np1-np4: https://github.com/srina1h/Tensor_Contract_Optim/tree/np1 ,[np2](https://github.com/srina1h/Tensor_Contract_Optim/tree/np2), [np3](https://github.com/srina1h/Tensor_Contract_Optim/tree/np3), [np4](https://github.com/srina1h/Tensor_Contract_Optim/tree/np4) - Individually profiling specific contractions

## Separate contraction programs:

Here different files are created on a single main branch. The files are explained below.

- contraction.py: https://github.com/srina1h/cuda_python/blob/main/contraction.py - Performing contractions on cuTensor using cuPy from python directly and profiling
- tdot.py: https://github.com/srina1h/cuda_python/blob/main/tdot.py - Performing contraction using tensordot and profilng
- contraction.cu: https://github.com/srina1h/cuda_python/blob/main/contraction.cu - Performing contractions on cuTensor directly using CUDA
- profile_everything.py: https://github.com/srina1h/cuda_python/blob/main/profile_everything.py - Profiling all algorithm variations in cuTensor through cuPy and tensordot as well

## Profiling result files:

The nsight systems profiling result files are stored here: https://github.com/srina1h/CSC591-EFTP/tree/main/profiling_results
