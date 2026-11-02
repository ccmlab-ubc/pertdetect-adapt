# pertdetect-adapt

Project code and data associated with the paper "Causal inference, prediction, and state estimation in sensorimotor learning" are provided in this repository. This work has been accepted for publication by *Proceedings of the Royal Society B* (doi: 10.1098/rspb.2025.1320). A preprint version of this work is accessible [here](https://www.biorxiv.org/content/10.1101/2025.01.30.635749v1). 


## Authors
---
**Hyosub E. Kim**, Romeo Chua, and Davin Hu

## Study summary
---
This study provides a normative explanation of how human observers effectively parse total movement error into internally- and externally-generated components for purposes of implicit sensorimotor recalibration.     


## Instructions
---
0) For access to the Python data science ecosystem, as used in this project, we recommend downloading with the free [Anaconda distribution](https://www.anaconda.com/download). 

1) To install the exact same Python packages and versions used for all analyses, you can recreate the same virtual environment by entering the following command in your terminal (within your cloned/downloaded project directory):

    `$ conda env create --name <nameyourenvironment> --file environment.yml`

2) These notebooks call on a package of custom-written functions found in the `src` directory. You may need to pip install this package. You can do so with the following command:

    `pip install -e .`


3) Once you've installed and conda activated the replicated environment, and pip installed the source code, you should be able to seamlessly run the Jupyter notebooks found in `experiment-1/scripts` to recreate the analyses:

    - `analyze-behavior.ipynb`  
    - `model-fitting.ipynb`  
    - `parameter-model-recovery.ipynb`    
     
4) The necessary data and results for running some parts of these notebooks are found in `experiment-1/results/`.

## Contact
---
For any questions about the study, please email Hyosub Kim (hyosub.kim@ubc.ca).












