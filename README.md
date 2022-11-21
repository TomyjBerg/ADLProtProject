# ADLProtProject


Instructions on creating virtual environment with all required packages: 
All explained here: https://www.youtube.com/watch?v=28eLP22SMTA&t=575s&ab_channel=PythonProgrammer
- Download python version 3.9.13, custom installation, save it somewhere at a defined place
- Create the venv in the folder where our project stuff is with path to the downloaded python 3.9.13
- Activate environment 
- Install all the required packages: 

pip install numpy
pip install -U matplotlib
pip install open3d
pip install -U scikit-learn
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-geometric

- In the console, type pip freeze to show all the packages that were installed and their versions, then you can copy the output of this to the requirements.txt textfile. (I can't do this as I have already installed some stupid packages in my venv that we will not need for our project)


This combination of pytorch 1.12.0 with pytorch geometric works fine for me, many other combinations have not worked. This is the cpu installation, later we could switch to the cuda installations for GPU. You can try it by running example_GNN
