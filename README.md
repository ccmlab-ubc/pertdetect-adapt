# pertdetect-adapt

Project code and data associated with the manuscript titled "Distinct computations underlie movement error parsing during implicit motor adaptation and perturbation detection". A preprint version of this work is accessible [here](https://www.biorxiv.org/content/10.64898/2026.06.25.734533v1). 


## Authors

**Hyosub E. Kim**, Jack O. Darley, Michael S. Landy, Romeo Chua, and Dusty Fox

## Study summary

This study combines behavioral and computational approaches towards understanding the relationship between implicit adaptation and explicit perturbation detection.  


## Instructions

0) For access to the Python data science ecosystem, as used in this project, we recommend downloading with the free [Anaconda distribution](https://www.anaconda.com/download). 

1) To install the exact same Python packages and versions used for all analyses, you can recreate the same virtual environment by entering the following command in your terminal (within your cloned/downloaded project directory):

    `$ conda env create --name <name-of-your-environment> --file environment.yml`

2) Activate the virtual environment.

    `$ conda active <name-of-your-environment>`

3) These notebooks call on a package of custom-written functions found in the `src` directory. You may need to pip install this package. You can do so with the following command:

    `pip install -e .`

4) Once you've installed and activated the replicated environment, and pip installed the source code, you should be able to seamlessly run the Jupyter notebooks found in `behavior` to recreate the analyses:

    - `analyze-behavior.ipynb`  
    - `model-fitting.ipynb`  
    - `parameter-model-recovery.ipynb`    
     
4) The necessary data and results for running some parts of these notebooks are found in `experiment-1/results/`.

## Contact

For any questions about the study, please email Hyosub Kim (hyosub.kim@ubc.ca).












