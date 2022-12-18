# ADLProtProject

Student : 
- Graber David
- Berger Thomas

### Repository with code for predicting protein-protein interactions with geometric deep learning and graph neural networks.

--------------------------------------------------------------------------------------------------
### Objectives

Aim : Neural Network to classify pairs of  protein surfaces into binding and not binding

Model: Siamese Graph Convolutional Neural Network​

---------------------------------------------------------------------------------------------------
### Content

The Git Hub folder contains 4 class files :

- c_ProteinGraph.py : Class storing the data of a ply file, including coordinates of the graph nodes (pos), 
    the features (x), and the adjacency information (edge_index)

- c_GraphPatch.py : Class storing the data of an extracted surface patch

- c_PatchDataset_sparse.py : Custom dataset for generation of datasets of graphs extracted from protein surfaces
returning the Patch with feature / edge index / label

- c_PatchDataset.py : Custom dataset for generation of datasets of graphs extracted from protein surfaces.
returning the Patch with feature / adjency matrix / label


The Git Hub folder contains 3 function files :

- f_extract_patches.py : functions related to the main functions "extract_patches" :

    1- taking a complex of the proteine, and extract the intacting paches from the center_iface node of each protein forming the complex.  

    2- extract two patches from a random center points of proteins inside the complex.

    The patches extracted are saved into the GraphPatch Class with the features, adgency matrix and edge index as arguement and label 1 if the patches are extracted from the interface 0 otherwise

- f_vizualize_pointcloud.py : Plot 3D pointcloud with Open3d.

- helper_fucntions.py : Fucntions used to save the data into pickle file and load a pickle file


The Git hub Folder contains differents notebook :

- 3Dplot.ipynb : Plot the distance between the average of each features for each edge pairs.

- parse_ply_files.ipynb : Read PLY file and translate data to a Graph. This grpah is saved as a python object and saved as a pickle file

- find_iface_points.ipynb : Take the Graph python object of a protein complex (from a pickle file) and its 2 subunits and find the index of the points in the subunits that corresponds to the interacting region. Retun indexes, positions and the center of the interface region for each subunit and added into the pickle file

- extract_pacthes.ipynb : Extraction of the pacthes of the proteingraph object using the "extract_pacthes" function and save thoses patches into GraphPatch class.

- model_siamese_baseline_small.ipynb : Perform Siamese Graph Neuron Network with a Global Mean Pooling on the GraphPatch Dataset.

- model_siamese_diff_pool : Perform Siamese Graph Neuron Network with a DiffPool on the GraphPatch Dataset.

---------------------------------------------------------------------------------------------------
### Dataset

Training Data: Compiled a dataset of protein complexes from the protein data bank to train the model.​

Data Generation : 

- Protein Surfaces as triangular meshes are generated with MSMS program and PyMesh.​ 

- Electrostatics computed with PDB2PQR and APBS​

- Assignment of hydropathy and electron/proton donor potential according to the closest atoms​


---------------------------------------------------------------------------------------------------
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


------------------------------------------------------------------------------------------------------
