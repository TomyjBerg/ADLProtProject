# ADLProtProject
Repository with code for predicting protein-protein interactions with geometric deep learning and graph neural networks. 


### Instructions on creating virtual environment with all required packages: 

- Download python version 3.9.13, custom installation, save it somewhere at a defined place
- Create the venv in the folder where the project files are with path to the downloaded python 3.9.13
- Activate environment 

Install all the required packages: 

- pip install numpy
- pip install -U matplotlib
- pip install -U scikit-learn
- pip install pandas
- pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
- pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
- pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
- pip install torch-geometric
- pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

In the console, type pip freeze to show all the packages that were installed and their versions, copy the output of this to the requirements.txt textfile.

This combination of pytorch 1.12.0 with pytorch geometric works fine for me, many other combinations have not worked. This is the cpu installation, later we could switch to the cuda installations for GPU. You can try it by running example_GNN


### When training models on GPU conda version 11.6 (to get cuda version: nvcc --version)

- pip install numpy
- pip install -U matplotlib
- pip install -U scikit-learn
- pip install pandas
- pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
- pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
- pip install torch-geometric
- pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
