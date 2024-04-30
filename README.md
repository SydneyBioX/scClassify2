# scClassify2
![](https://github.com/Wenze18/scClassify2/blob/main/scClassify2.png)

We proposed scClassify2, a novel framework for adjacent cell state identification with a transferable model across platforms with three key innovations:  

(1)   we use a dual layer architecture to enable joint learning from both the expression information of single cells and big-data-based gene co-expression patterns. We achieved this through a message passing neural network which captures two layers of information through edges and nodes representation,  

(2)   we adopt ordinal regression as the classifier of the network with a conditional training procedure specifically designed to identify adjacent cell state transition,  

(3)   we utilise log-ratio of expression values to capture the relationship between two genes. The log-ratio is shown to be more stable and hence more generalisable across datasets compared to individual gene expression.  


## Table of Contents

* [Installation](#Installation)
* [Usage](#Usage)
* [WebAPP](#WebAPP)
* [Citation](#Citation)

## Installation

To correctly use **scClassify2** via your local device, we suggest first create a conda environment by:

~~~shell
conda create --name <env> --file environment.yaml
conda activate <env>
~~~

Once success, you have the right environment to use scClassify2.
