import json,argparse,random,os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from torchviz import make_dot
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
import logging
import time
from data import prepare_data
from utils import Data_using
from model import scMPNN
from loss_function import CornLoss, SinkhornDistance
import os


def main(args):
    logging.basicConfig(level=logging.INFO,
                    filename='log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)

    lr = 1e-4
    batch_size = args.batch_size
    n_epoch = 28

    var_mata, using_data, first_level_train, second_level_data, second_level_label, gene_dict = prepare_data(args.trainset_path, args.tree_path)
    
    # first level training
    os.makedirs('./models/1st_level', exist_ok=True)
    train_data_use=first_level_train.X.toarray()
    gene_vecs=[]
    for gene_name in var_mata['GeneSymbol']:
        gene_vecs.append(gene_dict[gene_name.upper()])
    train_label=first_level_train.obs['1st_level']
    
    train_label_dict={cb:cb for cb in set(train_label)}
    trainset=Data_using(gene_vecs, train_data_use, train_label, train_label_dict, training = True)
    dataloader_train=DataLoader(dataset=trainset,batch_size=batch_size,
                                shuffle=True,num_workers=4)
    
    MPNNmodel= scMPNN(len(train_label_dict.keys()), len(gene_vecs), gene_vecs[0].shape[-1], 128, 64, 128, 1, 1, 8, 8).cuda()

    optimizer = optim.Adam(MPNNmodel.parameters(), lr=lr)
    step_size = 10  # Adjust this as needed
    gamma = 0.5  # The factor by which to reduce the learning rate
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    loss_class = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    loss_recover_distance = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
    
    for p in MPNNmodel.parameters():
        p.requires_grad = True
        
    for epoch in range(n_epoch):
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr'] 
        print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")
        
        MPNNmodel.train()
        loss_class_train=[]
        recovery_distance_train=[] 
        logger.info("--------****--------")
        logger.info('epoch: %d,' % (epoch))
        for index,item in enumerate(dataloader_train):            
            cur_gene, cur_cell, mask, classlabel = item
            MPNNmodel.zero_grad()
            
            cur_gene=torch.tensor(cur_gene).to(torch.float32).cuda()
            cur_cell=torch.tensor(cur_cell).to(torch.float32).cuda()
            mask=torch.tensor(mask).to(torch.float32).cuda()
            classlabel=torch.tensor(classlabel).cuda()
            recover, cell_state_logits = MPNNmodel(cur_gene, cur_cell, mask)
            
            err_class = loss_class(cell_state_logits, classlabel)*5
            dist_recover, P, C= loss_recover_distance(recover.view((-1,gene_vecs[0].shape[-1]))[(1-mask).view(-1).bool()],
                                                cur_gene.view((-1,gene_vecs[0].shape[-1]))[(1-mask).view(-1).bool()])
            dist_recover = dist_recover/10 
            loss= err_class + dist_recover 
            loss.backward()
            optimizer.step()

            loss_class_train.append(err_class.cpu().data.numpy())
            recovery_distance_train.append(dist_recover.cpu().data.numpy())
        torch.save(MPNNmodel, './models/1st_level/epoch_{0}.pth'.format(epoch))
        logger.info('epoch: %d, Train, loss_class: %f , recovery_dist: %f' \
                % (epoch,np.array(loss_class_train).mean(),np.array(recovery_distance_train).mean()))
        logger.info("--------****--------")
    
    # second level training    
    for branch in second_level_data.keys():
        logger.info("--------****--------")
        logger.info("--------second level: branch %d--------" % (branch))
        logger.info("--------****--------")
        os.makedirs('./models/2nd_level/branch_'+str(branch), exist_ok=True)
        train_data_use=second_level_data[branch].X.toarray()
        gene_vecs=[]
        for gene_name in var_mata['GeneSymbol']:
            gene_vecs.append(gene_dict[gene_name.upper()])
        train_label=second_level_data[branch].obs['train_data$CellType']   
        train_label_dict=second_level_label[branch]
        trainset=Data_using(gene_vecs, train_data_use, train_label, train_label_dict, training = True)
        dataloader_train=DataLoader(dataset=trainset,batch_size=batch_size,
                                shuffle=True,num_workers=4)
    
        MPNNmodel= scMPNN(len(train_label_dict.keys()), len(gene_vecs), gene_vecs[0].shape[-1], 128, 64, 128, 1, 1, 8, 8).cuda()
        optimizer = optim.Adam(MPNNmodel.parameters(), lr=lr)
        step_size = 10  # Adjust this as needed
        gamma = 0.5  # The factor by which to reduce the learning rate
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
        loss_class = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
        loss_recover_distance = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
    
        for p in MPNNmodel.parameters():
            p.requires_grad = True
        
        for epoch in range(n_epoch):
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr'] 
            print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")
            
            MPNNmodel.train()
            loss_class_train=[]
            recovery_distance_train=[] 
            logger.info("--------****--------")
            logger.info('epoch: %d,' % (epoch))
            for index,item in enumerate(dataloader_train):            
                cur_gene, cur_cell, mask, classlabel = item
                MPNNmodel.zero_grad()
                
                cur_gene=torch.tensor(cur_gene).to(torch.float32).cuda()
                cur_cell=torch.tensor(cur_cell).to(torch.float32).cuda()
                mask=torch.tensor(mask).to(torch.float32).cuda()
                classlabel=torch.tensor(classlabel).cuda()
                recover, cell_state_logits = MPNNmodel(cur_gene, cur_cell, mask)
                
                err_class = loss_class(cell_state_logits, classlabel)*5
                dist_recover, P, C= loss_recover_distance(recover.view((-1,gene_vecs[0].shape[-1]))[(1-mask).view(-1).bool()],
                                                    cur_gene.view((-1,gene_vecs[0].shape[-1]))[(1-mask).view(-1).bool()])
                dist_recover = dist_recover/10 
                loss= err_class + dist_recover 
                loss.backward()
                optimizer.step()

                loss_class_train.append(err_class.cpu().data.numpy())
                recovery_distance_train.append(dist_recover.cpu().data.numpy())
            torch.save(MPNNmodel, './models/2nd_level/branch_'+str(branch)+'/epoch_{0}.pth'.format(epoch))
            logger.info('epoch: %d, Train, loss_class: %f , recovery_dist: %f' \
                    % (epoch,np.array(loss_class_train).mean(),np.array(recovery_distance_train).mean()))
            logger.info("--------****--------")
        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trainset_path', type=str, default='./train_data.h5ad')
    parser.add_argument('--tree_path', type=str, default='./tree.txt')
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    main(args)
