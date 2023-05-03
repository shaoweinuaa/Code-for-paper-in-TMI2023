from __future__ import division
from __future__ import print_function
import os
import argparse
import math
import torch.nn as nn


from DGLGAT import GatNet1
import torch
import pandas as pd
import pickle
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from selection import models, data
from selection.models import utils
import random
import numpy as np
from torch.backends import cudnn
import dgl
from TCGADataset import TCGA_dataset

from attention.attention import fusion
from typing import  Callable
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader

def GCCA_loss(H_list):
    r = 1e-4
    eps = 1e-8

    top_k = 10

    AT_list = []

    for H in H_list:

        assert torch.isnan(H).sum().item() == 0

        o_shape = H.size(0) 
        m = H.size(1) 

        Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(Hbar).sum().item() == 0

        A, S, B = Hbar.svd(some=True, compute_uv=True)

        A = A[:, :top_k]

        assert torch.isnan(A).sum().item() == 0

        S_thin = S[:top_k]

        S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)

        assert torch.isnan(S2_inv).sum().item() == 0

        T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)

        assert torch.isnan(T2).sum().item() == 0

        T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device).float())

        T = torch.diag(torch.sqrt(T2))

        assert torch.isnan(T).sum().item() == 0

        T_unnorm = torch.diag(S_thin + eps)

        assert torch.isnan(T_unnorm).sum().item() == 0

        AT = torch.mm(A, T)
        AT_list.append(AT)

    M_tilde = torch.cat(AT_list, dim=1)

    assert torch.isnan(M_tilde).sum().item() == 0

    _, S, _ = M_tilde.svd(some=True)

    assert torch.isnan(S).sum().item() == 0
    use_all_singular_values = False
    if not use_all_singular_values:
        S = S[:top_k]

    corr = torch.sum(S)
    assert torch.isnan(corr).item() == 0

    loss = - corr
    return loss

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)  
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata) 
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1 
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx] 
    T2 = survtime_all[~idx] 
    E1 = labels[idx] 
    E2 = labels[~idx]  
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]:
                        concord = concord + 1
                    elif hazards[j] < hazards[i]:
                        concord = concord + 0.5

    return (concord / total)

def AUC(Hazard,Status,Time):
    total=0
    correct=0
    for i in range(len(Time)):
        if Time[i]!=max(Time) and Time[i]!=min(Time):
            for j in range(len(Time)):
                for k in range(len(Time)):
                    if j!=i and k!=i:
                        if Status[j]==1 and Time[j]<Time[i] and Time[k]>Time[i]:
                            total+=1
                            if Hazard[j]>Hazard[k]:
                                correct+=1
    return correct/total

def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.cpu().numpy()
    return (concordance_index(survtime_all, -hazards, labels))

def test(model,fsmodel,fsmodelmiRNA, sa,layer1,datasets,verbose):
    with torch.no_grad():
        lbl_pred_all = None
        lbl_all = None
        survtime_all = None
        code_final = None
        loss_nn_sum = 0
        model.eval()
        fsmodel.eval()
        fsmodelmiRNA.eval()
        sa.eval()
        layer1.eval()
        iter = 0
        for tags, lbl, survtime, gs, mrna, mirna in datasets:
            lbl.to(args.device)
            survtime.to(args.device)
            fs = mrna
            fs = maxmin(fs)  

            fs1 = mirna
            fs1 = maxmin(fs1)  

            gs = gs.to(args.device)
            g, code, lbl_pred, atten_final, edges = model(gs)

            T, featureGene = fsmodel(fs.float()) 
            T1, featureGene1 = fsmodelmiRNA(fs1.float())  

            Hlist = []
            Hlist.append(code)
            Hlist.append(featureGene)
            Hlist.append(featureGene1)

            DGCCA_loss = GCCA_loss(Hlist)

            output = sa(code, featureGene, featureGene1)
            lbl_pred = layer1(output)

            if iter == 0:
                lbl_pred_all = lbl_pred
                lbl_all = lbl
                survtime_all = survtime
                code_final = code
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_all = torch.cat([lbl_all, lbl])
                survtime_all = torch.cat([survtime_all, survtime])
                code_final = torch.cat([code_final, code])

            current_batch_len = len(survtime)
            R_matrix_test = np.zeros([current_batch_len, current_batch_len], dtype=int)
            for i in range(current_batch_len):
                for j in range(current_batch_len):
                    R_matrix_test[i, j] = survtime[j] >= survtime[i]

            test_R = torch.FloatTensor(R_matrix_test)

            test_R = test_R.to(args.device)
            test_ystatus = lbl

            theta = lbl_pred.reshape(-1)
            exp_theta = torch.exp(theta)

            loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * test_R, dim=1))) * test_ystatus.float())
            loss_nn = loss_nn - DGCCA_loss * 1e-3
            loss_nn_sum = loss_nn_sum + loss_nn.data.item()

            iter += 1

        code_final_4_original_data = code_final.data.cpu().numpy()


        acc_test = accuracy_cox(lbl_pred_all.data, lbl_all)
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
        c_index = CIndex(lbl_pred_all.data, lbl_all, survtime_all)
        auc = AUC(lbl_pred_all.data, lbl_all, survtime_all)

    if verbose > 0:
        print('\n[Testing]\t loss (nn):{:.4f}'.format(loss_nn_sum),'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

    return (code_final_4_original_data, loss_nn_sum, acc_test, \
            pvalue_pred, c_index, lbl_pred_all.data.cpu().numpy().reshape(-1), lbl_all, survtime_all,auc)

def val(model, fsmodel, fsmodelmiRNA, sa, layer, datasets, verbose):
    with torch.no_grad():
        lbl_pred_all = None
        lbl_all = None
        survtime_all = None
        code_final = None
        loss_nn_sum = 0
        model.eval()
        fsmodel.eval()
        fsmodelmiRNA.eval()
        sa.eval()
        layer.eval()
        iter = 0
        for tags, lbl, survtime, gs, mrna, mirna  in datasets:
            lbl.to(args.device)
            survtime.to(args.device)
            fs = mrna
            fs = maxmin(fs) 

            fs1 = mirna
            fs1 = maxmin(fs1)  

            gs = gs.to(args.device)
            g, code, lbl_pred, atten_final, edges = model(gs)

            T, featureGene = fsmodel(fs.float())  
            T1, featureGene1 = fsmodelmiRNA(fs1.float())  

            Hlist = []
            Hlist.append(code)
            Hlist.append(featureGene)
            Hlist.append(featureGene1)

            DGCCA_loss = GCCA_loss(Hlist)

            output = sa(code, featureGene, featureGene1)
            lbl_pred = layer(output)

            if iter == 0:
                lbl_pred_all = lbl_pred
                lbl_all = lbl
                survtime_all = survtime
                code_final = code
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_all = torch.cat([lbl_all, lbl])
                survtime_all = torch.cat([survtime_all, survtime])
                code_final = torch.cat([code_final, code])

            current_batch_len = len(survtime)
            R_matrix_test = np.zeros([current_batch_len, current_batch_len], dtype=int)
            for i in range(current_batch_len):
                for j in range(current_batch_len):
                    R_matrix_test[i, j] = survtime[j] >= survtime[i]

            test_R = torch.FloatTensor(R_matrix_test)
            test_R = test_R.to(args.device)
            test_ystatus = lbl
            theta = lbl_pred.reshape(-1)
            exp_theta = torch.exp(theta)
            loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * test_R, dim=1))) * test_ystatus.float())- DGCCA_loss * 1e-3
            loss_nn_sum = loss_nn_sum + loss_nn.data.item()
            iter += 1

        code_final_4_original_data = code_final.data.cpu().numpy()
        acc_test = accuracy_cox(lbl_pred_all.data, lbl_all)
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
        c_index = CIndex(lbl_pred_all.data, lbl_all, survtime_all)

    if verbose > 0:
        print('\n[Validation]\t loss (nn):{:.4f}'.format(loss_nn_sum),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

    return (code_final_4_original_data, loss_nn_sum, acc_test, \
            pvalue_pred, c_index, lbl_pred_all.data.cpu().numpy().reshape(-1), lbl_all, survtime_all)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def Seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    dgl.seed(seed)

def maxmin(tensor):
    max = torch.max(tensor)
    min = torch.min(tensor)
    return (tensor-max)/(max-min)

class Linear(nn.Module): 
    def __init__(self,in_features, out_features):
        super(Linear,self).__init__()
        self.w = nn.Parameter(torch.randn(in_features,out_features)).to(args.device)
        self.b = nn.Parameter(torch.randn(out_features)).to(args.device)

    def forward(self, x):
        x = x.mm(self.w)    
        x = maxmin(x)
        return x+self.b.expand_as(x)

if __name__ == "__main__":
            parse = argparse.ArgumentParser()
            parse.add_argument('--weight_decay', type=float, default=0, help='weight decay')
            parse.add_argument('--lr', type=float, default=0.001,help='learning rate')
            parse.add_argument('--batch_size', type=int, default= 64 ,help='batch size')
            parse.add_argument('--epochs', type=int, default= 50 , help='maximum number of epochs')
            args = parse.parse_args()
            args.device = 'cpu'
            iter1 = 0
            for sheetname in {'Fold1','Fold2','Fold3','Fold4','Fold5'}:
                Seed(666)
                df1 = pd.read_excel('BRCA_Dataset.xlsx', sheet_name=sheetname)
                list_file = open('BRCA/BRCA_dataset.pickle', 'rb')
                loader = pickle.loads(list_file.read())

                trainlist = []
                validationlist = []
                testlist = []
                for index, row in df1.iterrows():
                    ID = row['ID']
                    value = row['Type']
                    if value == 'training':
                            trainlist.append(ID)
                    if value == 'validation':
                            validationlist.append(ID)
                    if value == 'test':
                            testlist.append(ID)

                training_set = []
                validation_set = []
                testing_set = []
                train_loader = []
                val_loader =  []
                test_loader = []
                tagslist = loader.tag

                for train_sample in trainlist:
                    for tags, lbl, survtime, gs, mrna, mirna in loader:
                        if str(train_sample) == str(tags):
                            training_set.append(tagslist.index(tags))
                for test_sample in testlist:
                    for tags, lbl, survtime, gs, mrna, mirna in loader:
                        if str(test_sample) == str(tags):
                            testing_set.append(tagslist.index(tags))
                for val_sample in validationlist:
                    for tags, lbl, survtime, gs, mrna, mirna in loader:
                        if str(val_sample) == str(tags):
                            validation_set.append(tagslist.index(tags))

                for index in training_set:
                    train_loader.append(loader[index])
                for index in testing_set:
                    test_loader.append(loader[index])
                for index in validation_set:
                    val_loader.append(loader[index])

                train_loader = GraphDataLoader(train_loader, batch_size=args.batch_size, shuffle=True)
                val_loader = GraphDataLoader(val_loader, batch_size=args.batch_size, shuffle=True)
                test_loader = GraphDataLoader(test_loader, batch_size=args.batch_size, shuffle=True)

                model = GatNet1(
                    input=64,
                    hiddens=[[256, 1, 0.8], [512, 1, 0.8]],
                    classifier=nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
                ).to(args.device)

                fsmodel = models.SelectorMLP(  
                    input_layer='concrete_selector',
                    k=10,
                    input_size=57,
                    output_size=1,
                    hidden=[512, 512],
                    activation='elu').to(args.device)

                fsmodelmiRNA = models.SelectorMLP( 
                    input_layer='concrete_selector',
                    k=10,
                    input_size=12,
                    output_size=1,
                    hidden=[512, 512],
                    activation='elu').to(args.device)

                sa = fusion(512)

                layer = Linear(512, 1).to(args.device)

                warm_up_epochs = 10
                train_params = [
                    {'params':model.parameters(),'lr':args.lr},
                    {'params':fsmodel.parameters(),'lr':args.lr},
                    {'params':sa.parameters(),'lr':args.lr},
                    {'params':layer.parameters(),'lr':args.lr},
                    {'params':fsmodelmiRNA.parameters(), 'lr': args.lr}]
                optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
                warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
                        math.cos((epoch - warm_up_epochs) / (num_epochs - warm_up_epochs) * math.pi) + 1)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

                c_index_list = {}
                c_index_list['train'] = []
                c_index_list['test'] = []
                loss_nn_all = []
                pvalue_all = []
                c_index_all = []
                acc_train_all = []
                c_index_best = 0
                c_index_val_best = 0
                c_index_new = 0
                code_output = None
                verbose = 1
                measure = True
                num_epochs = args.epochs

                epoch_number = 1
                for epoch in range(args.epochs):
                    model.train()
                    fsmodel.train()
                    sa.train()
                    layer.train()
                    fsmodelmiRNA.train()
                    lbl_pred_all = None
                    lbl_all = None
                    survtime_all = None
                    code_final = None
                    loss_nn_sum = 0
                    iter = 0
                    for tags, lbl, survtime, gs, mrna, mirna  in train_loader:
                        optimizer.zero_grad() 
                        lbl.to(args.device)
                        survtime.to(args.device)

                        fs = mrna
                        fs = maxmin(fs)

                        fs1 = mirna
                        fs1 = maxmin(fs1)

                        gs = gs.to(args.device)
                        g, code, lbl_pred, atten_final, edges = model(gs)


                        genefeaturematrix = fs

                        T,featureGene = fsmodel(genefeaturematrix.float()) 
                        T1,featureGene1 = fsmodelmiRNA(fs1.float()) 
                        Hlist = []
                        Hlist.append(code)
                        Hlist.append(featureGene)
                        Hlist.append(featureGene1)

                        DGCCA_loss = GCCA_loss(Hlist)

                        output = sa(code,featureGene,featureGene1)
                        lbl_pred = layer(output)

                        if iter == 0:
                            lbl_pred_all = lbl_pred
                            survtime_all = survtime
                            lbl_all = lbl
                            code_final = output

                        else:
                            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                            lbl_all = torch.cat([lbl_all, lbl])
                            survtime_all = torch.cat([survtime_all, survtime])
                            code_final = torch.cat([code_final, output]) 

                        current_batch_len = len(survtime)
                        R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
                        for i in range(current_batch_len):
                            for j in range(current_batch_len):
                                R_matrix_train[i, j] = survtime[j] >= survtime[i]

                        train_R = torch.FloatTensor(R_matrix_train)
                        train_R = train_R.to(args.device)
                        train_ystatus = lbl
                        theta = lbl_pred.reshape(-1)
                        exp_theta = torch.exp(theta)
                        loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus.float())
                        loss_nn = loss_nn - DGCCA_loss * 1e-3
                        loss = loss_nn
                        loss_nn_sum = loss_nn_sum + loss_nn.data.item()
                        # ===================backward====================
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        iter += 1

                    code_final_4_original_data = code_final.data.cpu().numpy()
                    survival_final_4_original_data = survtime_all.data.cpu().numpy()
                    if measure or epoch == (num_epochs - 1):
                        acc_train = accuracy_cox(lbl_pred_all.data, lbl_all)
                        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
                        c_index = CIndex(lbl_pred_all.data, lbl_all, survtime_all)
                        c_index_list['train'].append(c_index)


                        if verbose > 0:
                            print('\nEpoch:{}\t[Training]\t loss (nn):{:.4f}'.format(epoch_number,loss_nn_sum),'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
                            with open("BRCA_TMI.txt", "a") as f:
                                f.write('\nEpoch:{}\t[Training]\t loss (nn):{:.4f},c_index: {:.4f}, p-value: {:.3e}'.format(epoch_number,loss_nn_sum,c_index, pvalue_pred))

                        pvalue_all.append(pvalue_pred)
                        c_index_all.append(c_index)
                        loss_nn_all.append(loss_nn_sum)
                        acc_train_all.append(acc_train)


                        code_validation, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all, OS_event, OS = val(model,fsmodel,fsmodelmiRNA,sa,layer,val_loader,verbose)

                        if c_index_pred > c_index_val_best:
                            with open("BRCA_TMI.txt", "a") as f:
                                f.write('\n[Validation]\t loss (nn):{:.4f},c_index: {:.4f}, p-value: {:.3e}'.format(loss_nn_sum,c_index_pred,pvalue_pred))
                            c_index_val_best = c_index_pred
                            code_validation, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all, OS_event, OS,aucvalue = test(model,fsmodel,fsmodelmiRNA,sa,layer,test_loader,verbose)

                            with open("BRCA_TMI.txt", "a") as f:
                                f.write('\n[Testing]\t loss (nn):{:.4f},C_Index: {:.4f},AUC: {:.4f},p-value: {:.3e} '.format(loss_nn_sum,c_index_pred,aucvalue,pvalue_pred))
                            c_index_test = c_index_pred
                        utils.input_layer_fix(fsmodel.input_layer)
                        indsM = fsmodel.get_inds()

                        utils.input_layer_fix(fsmodelmiRNA.input_layer)
                        indsMI = fsmodelmiRNA.get_inds()

                    epoch_number = epoch_number + 1
                with open("BRCA_TMI.txt", "a") as f:
                    f.write("\nC_index:{:.4f},AUC{:.4f}：".format(c_index_test, aucvalue))

