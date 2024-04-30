# scClassify2
![](https://github.com/Wenze18/scClassify2/blob/main/scClassify2.png)

We proposed scClassify2, a novel framework for adjacent cell state identification with a transferable model across platforms with three key innovations:  

(1)   we use a dual layer architecture to enable joint learning from both the expression information of single cells and big-data-based gene co-expression patterns. We achieved this through a message passing neural network which captures two layers of information through edges and nodes representation,  

(2)   we adopt ordinal regression as the classifier of the network with a conditional training procedure specifically designed to identify adjacent cell state transition,  

(3)   we utilise log-ratio of expression values to capture the relationship between two genes. The log-ratio is shown to be more stable and hence more generalisable across datasets compared to individual gene expression.  


## Table of Contents

* [Installation&Usage](#Installation&Usage)
* [WebAPP](#WebAPP)
* [Citation](#Citation)

## Installation&Usage

To correctly use **scClassify2** via your local device, we suggest first create a conda environment by:

~~~shell
conda create --name <env> --file environment.yaml
conda activate <env>
~~~

Once success, you have the right environment to use scClassify2.  
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

## WebAPP
To provide a resource for the scientific community, we developed a user-friendly web server called [scClassify-catalogue](https://shiny.maths.usyd.edu.au/scClassify_catalogue/), offering a comprehensive catalogue of scClassify2 models trained from datasets covering almost 1,000 cell types and 30 tissue types.  

scClassify-catalogue directly delivers annotation results for the selected tissue within a relatively short time. The input of scClassify-catalogue is flexible and allows either the direct output from 10x cellranger software, or a csv file or h5ad file containing the gene expression matrix. Once the job is submitted, a job ID will be made available. A file containing the predicted outcome as well as an HTML file will be emailed to the email address entered by the user once the job finishes. The HTML file visualises the predicted cell type such that the user can easily check the prediction result.   

![]()

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
