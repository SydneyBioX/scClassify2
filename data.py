import scanpy as sc, numpy as np, pandas as pd, anndata as ad


def prepare_data(data_path): 
    #### read raw data
    adata = sc.read(data_path)
    data=adata
    sc.pp.filter_cells(data, min_genes=200)
    sc.pp.filter_genes(data, min_cells=10)
    sc.pp.highly_variable_genes(data, n_top_genes = 600, subset = True, flavor='seurat_v3') ###v3 is the only one for count data
    
    #### read the gene2vec dict
    with open('./gene2vec_dim_200_iter_9.txt','r') as f:
        contents=f.readlines()
    gene_dict={}
    for line in contents:
        numbers=line.split()
        gene_dict[numbers[0]]=np.array([float(numbers[i]) for i in range(1,len(numbers))])
    gene_ifhasrecord=[i.upper() in gene_dict.keys() for i in data.var['feature_id']]
    using_data=data[:,gene_ifhasrecord]
    cell_indices = np.arange(using_data.n_obs)
    np.random.shuffle(cell_indices)
    using_data = using_data[cell_indices, :]
    using_data.obs['CellState']=using_data.obs['cell_type']
    var_mata=using_data.var.copy()
    var_mata['GeneSymbol']=var_mata['feature_id']

    #### prepare for 5 fold valid
    amounts=[]
    subsets=[]
    for i in care:
        cur=using_data[using_data.obs['CellState']==i]
        subsets.append(cur)   
        amounts.append(cur.X.shape[0])
        
    train_set=[]
    validation_set=[]
    for i in range(5):
        fold_i=[]
        remains_i=[]
        for j in range(len(subsets)):
            fold_i.append(subsets[j][(amounts[j]//5)*i:(amounts[j]//5)*(i+1)])
            if i==0:
                remains_i.append(subsets[j][(amounts[j]//5):])
            elif i==4:
                remains_i.append(subsets[j][:(amounts[j]//5)*4])
            else:
                remains_i.append(sc.concat([ subsets[j][:(amounts[j]//5)*i], subsets[j][(amounts[j]//5)*(i+1):] ], index_unique=None, join="outer"))
                           
        train_set.append(sc.concat(remains_i, index_unique=None, join="outer"))
        validation_set.append(sc.concat(fold_i, index_unique=None, join="outer"))
    
    return var_mata, using_data, train_set, validation_set, gene_dict