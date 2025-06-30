# scClassify2
| ![](https://github.com/Wenze18/scClassify2/blob/main/scClassify2.png) |
|:-----------------------------------:|
| **Figure 1. **  Overview of the scClassify2 framework for sequential cell state identification. |

We proposed scClassify2, a novel framework for adjacent cell state identification with a transferable model across platforms with three key innovations:  

(1)   we use a dual layer architecture to enable joint learning from both the expression information of single cells and big-data-based gene co-expression patterns. We achieved this through a message passing neural network which captures two layers of information through edges and nodes representation,  

(2)   we adopt ordinal regression as the classifier of the network with a conditional training procedure specifically designed to identify adjacent cell state transition,  

(3)   we utilise log-ratio of expression values to capture the relationship between two genes. The log-ratio is shown to be more stable and hence more generalisable across datasets compared to individual gene expression.  

The network architecture is:
![](https://github.com/Wenze18/scClassify2/blob/main/network_architecture.png)
*Figure 2. The network architecture of scClassify2 *

## Table of Contents

* [Installation](#Installation&Usage)
* [Usage](#Usage)
* [WebAPP](#WebAPP)
* [Citation](#Citation)

## Installation

To correctly use **scClassify2** via your local device, we suggest first create a conda environment by:

~~~shell
conda create -n scclassify2 python=3.9
conda activate scclassify2
conda install -c conda-forge anndata
conda install -c conda-forge scanpy
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install wandb
pip install torchviz
pip install --user scikit-misc
~~~

Note, as we tested on different platforms, sometimes you may need to reinstall h5py package by:
~~~shell
pip uninstall h5py
pip install h5py
~~~

Once success, you have the right environment to use scClassify2.

## Usage

Before actual training process, you need to change some paths of input files. You can either go into the main.py file to change arguments in lines
~~~
parser.add_argument('--trainset_path', type=str, default='./train_set.h5ad')
parser.add_argument('--label_path', type=str, default='./label.json')
~~~
according to comments of corresponding codes, or directly pass these paths when invoking the program by
~~~
python main.py --trainset_path <YourPath> --label_path <YourPath>
~~~
scClassify2 also keep the hierarchical identification property of scClassify by using HOPACK tree, details could be found in the folder "code_with_HOPACK".

### 1.try with demo data
If you want to have a quick experience of scClassify2 traning, then directly use this:
~~~shell
conda activate scclassify2  
python main.py --trainset_path ./train_set.h5ad --label_path ./label.json 
~~~
### 2.try with demo custom data
If you want to try your own dataset, then you must prepare you dataset first. <br><br>
a. Convert your data into H5AD format. The H5AD format is the HDF5 disk representation commonly used to share single-cell datasets. Both R and Python have many packages for users to easily transfer their data into H5AD format. During the process, you must store the cell state lables into "CellState" column. Name the converted file as input.h5ad and copy it into current path (the scClassiy2 folder).<br><br>
b. Prepare cell state labels as a list and stored it into a JSON file (please check ./label.json file as template). Name the JSON file as input.json and copy it into current path..<br><br>
The use this:
~~~shell
conda activate scclassify2  
python main.py --trainset_path ./input.h5ad --label_path ./input.json 
~~~

## WebAPP
To provide a resource for the scientific community, we developed a user-friendly web server called [scClassify-catalogue](https://shiny.maths.usyd.edu.au/scClassify_catalogue/), offering a comprehensive catalogue of scClassify2 models trained from datasets covering almost 1,000 cell types and 30 tissue types.  

scClassify-catalogue directly delivers annotation results for the selected tissue within a relatively short time. The input of scClassify-catalogue is flexible and allows either the direct output from 10x cellranger software, or a csv file or h5ad file containing the gene expression matrix. Once the job is submitted, a job ID will be made available. A file containing the predicted outcome as well as an HTML file will be emailed to the email address entered by the user once the job finishes. The HTML file visualises the predicted cell type such that the user can easily check the prediction result.   

![](https://github.com/Wenze18/scClassify2/blob/main/Server.png)
*Figure 2. Overview of scClassify-catalogue - the web server *

## Citation

If you find our codes useful, please consider citing our work:

~~~bibtex


@article{scClassify2,
  title={A Message Passing Framework for Precise Cell State Identification with scClassify2},
  author={Wenze Ding, Yue Cao, Helen Fu, , Marni Torkel and Jean Yang},
  journal={},
  year={2024},
}
~~~
