import scanpy as sc, numpy as np, pandas as pd, anndata as ad

def prepare_data(data_path,tree_path): 
    #### read raw data
    adata = sc.read(data_path)
    data=adata
    
    with open('../gene2vec_dim_200_iter_9.txt','r') as f:
        contents=f.readlines()
    gene_dict={}
    for line in contents:
        numbers=line.split()
        gene_dict[numbers[0]]=np.array([float(numbers[i]) for i in range(1,len(numbers))])


    gene_ifhasrecord=[i.upper() in gene_dict.keys() for i in data.var['Symbol']]
    using_data=data[:,gene_ifhasrecord]
    cell_indices = np.arange(using_data.n_obs)
    np.random.shuffle(cell_indices)
    using_data = using_data[cell_indices, :]
    var_mata=using_data.var.copy()
    var_mata['GeneSymbol']=var_mata['Symbol']
    
    
    with open(tree_path, 'r') as ffff:
        contents_=ffff.readlines()
    tree={}
    care=[]
    for line in contents_:
        care.append(line.split(':')[0])
        if int(line.split(':')[1][:-1]) not in tree.keys():
            tree[int(line.split(':')[1][:-1])]=[] 
        tree[int(line.split(':')[1][:-1])].append(line.split(':')[0])
    
    change_1level={}
    for iii in tree.keys():
        for jjj in tree[iii]:
            change_1level[jjj]=iii

    
    first_level_train= using_data.copy()
    level1=[]
    for i in range(len(first_level_train.obs['train_data$CellType'])):
        level1.append(change_1level[first_level_train.obs['train_data$CellType'][i]])
    first_level_train.obs['1st_level']=level1
    
    second_level_data={}
    second_level_label={}    
    for abc in tree.keys():
        if len(tree[abc])>1:
            whether_this_type=[cur_obs in tree[abc] for cur_obs in using_data.obs['train_data$CellType']]
            cur_data= using_data.copy()
            second_level_data[abc]=cur_data[whether_this_type]
            second_level_label[abc]={k:i for (i,k) in enumerate(tree[abc])}
    
    return var_mata, using_data, first_level_train, second_level_data, second_level_label, gene_dict

