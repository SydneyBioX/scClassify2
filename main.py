import json,argparse,random,os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import wandb
from torchviz import make_dot
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
import logging
import time
from data import prepare_data
from utils import Data_using, get_prediction
from model import scMPNN
from loss_function import CornLoss, SinkhornDistance

def main(args):
    logging.basicConfig(level=logging.INFO,
                    filename='log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)

    lr = 1e-4
    batch_size = args.batch_size
    n_epoch = 28

    # load data into dataloader
    var_mata, using_data, train_set, validation_set, gene_dict = prepare_data(args.trainset_path)
    
    # load data label
    with open(args.label_path,'r') as f0:
        care=json.load(f0)
    label_dict={k:i for (i,k) in enumerate(care)}
    
    for fold in range(len(validation_set)):
        train_data_use=train_set[fold].X.toarray()
        test_data_use=validation_set[fold].X.toarray()
        gene_vecs=[]
        for gene_name in var_mata['GeneSymbol']:
            gene_vecs.append(gene_dict[gene_name.upper()])
        train_label=train_set[fold].obs['CellState']
        test_label=validation_set[fold].obs['CellState']
       
        trainset=Data_using(gene_vecs, train_data_use, train_label, label_dict, training = True)
        testset=Data_using(gene_vecs, test_data_use, test_label, label_dict, training = False,)
        dataloader_train=DataLoader(dataset=trainset,batch_size=batch_size,
                                    shuffle=True,num_workers=4)
        dataloader_test=DataLoader(dataset=testset,batch_size=batch_size,
                                    shuffle=True,num_workers=4)

        MPNNmodel= scMPNN(len(label_dict.keys()), len(gene_vecs), gene_vecs[0].shape[-1], 128, 64, 128, 1, 1, 8, 8).cuda()
        
        # setup optimizer
        optimizer = optim.Adam(MPNNmodel.parameters(), lr=lr)
        step_size = 10  # Adjust this as needed
        gamma = 0.5  # The factor by which to reduce the learning rate
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        loss_class = CornLoss(len(label_dict.keys()))
        loss_recover_distance = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        
        for p in MPNNmodel.parameters():
            p.requires_grad = True

        # training
        for epoch in range(n_epoch):
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr'] 
            print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")
            
            MPNNmodel.train()
            loss_class_train=[]
            recovery_distance_train=[] 
            logger.info("--------****--------")
            logger.info('epoch: %d,' % (epoch))
            time1=time.time()

            for index,item in enumerate(dataloader_train):
                
                cur_gene, cur_cell, mask, classlabel = item
                MPNNmodel.zero_grad()               
                cur_gene=torch.tensor(cur_gene).to(torch.float32).cuda()
                cur_cell=torch.tensor(cur_cell).to(torch.float32).cuda()
                mask=torch.tensor(mask).to(torch.float32).cuda()
                classlabel=torch.tensor(classlabel).cuda()
                recover, cell_state_logits = MPNNmodel(cur_gene, cur_cell, mask)
                
                err_class = loss_class(cell_state_logits, classlabel) *5
                dist_recover, P, C= loss_recover_distance(recover.view((-1,gene_vecs[0].shape[-1]))[(1-mask).view(-1).bool()],
                                                    cur_gene.view((-1,gene_vecs[0].shape[-1]))[(1-mask).view(-1).bool()])
                dist_recover = dist_recover/10 ### according to observation, makes two parts 1:1
                loss= err_class + dist_recover 
                loss.backward()
                optimizer.step()
                time2=time.time()

                loss_class_train.append(err_class.cpu().data.numpy())
                recovery_distance_train.append(dist_recover.cpu().data.numpy())
                

            logger.info('epoch: %d, Train, loss_class: %f , recovery_dist: %f' \
                    % (epoch,np.array(loss_class_train).mean(),np.array(recovery_distance_train).mean()))

            # validation
            loss_test=[]
            MPNNmodel.eval()
            num_testdata=0
            sum_testright=0
            for index,item in enumerate(dataloader_test):
                cur_gene, cur_cell, mask, classlabel = item 
                num_testdata += len(classlabel)
                cur_gene=torch.tensor(cur_gene).to(torch.float32).cuda()
                cur_cell=torch.tensor(cur_cell).to(torch.float32).cuda()
                mask=torch.tensor(mask).to(torch.float32).cuda()
                classlabel=torch.tensor(classlabel).cuda()
                recover, cell_state_logits = MPNNmodel(cur_gene, cur_cell, mask)
                predicted_labels = get_prediction(cell_state_logits)
                sum_testright += (predicted_labels == classlabel).sum().item()
                loss_test.append(loss_class(cell_state_logits, classlabel).cpu().data.numpy())


            logger.info('epoch: %d, Evaluation, err_class: %f, accuracy: %f,' \
                    % (epoch,np.array(loss_test).mean(), sum_testright/num_testdata))
            logger.info("--------****--------")
            

            torch.save(MPNNmodel, 'models/epoch_{0}.pth'.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## please replace this path with your own data path, and should convert it into a .h5ad file
    parser.add_argument('--trainset_path', type=str, default='./train_set.h5ad') 
    
    # data label should be a json file containing a list of your cell states, something like this: 
    # ['Proerythrocytes', 'Erythroblast', 'Erythrocytes']
    parser.add_argument('--label_path', type=str, default='./label.json')
    parser.add_argument('--hvg_num', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    main(args)
