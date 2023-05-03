# %%
from typing import Tuple, Callable
import numpy as np
import pandas as pd
import torch
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from tqdm import tqdm
import random
import pickle
from sklearn.decomposition import PCA

def rp(path: str) -> str:
 
    try:
        return os.path.join(os.path.dirname(__file__), path)
    except NameError:
        return path


def handleXls(excel_url: str, verbose=True) -> pd.DataFrame:
  
    if verbose:
        print("===================================")
  
    xls = pd.read_excel(excel_url)
    if verbose:
        print("读入", xls.shape[0], "行数据\n")
   
    xls['patient.bcr_patient_barcode'] = xls['patient.bcr_patient_barcode'].str.upper()

    
    if verbose:
        print("生存情况的处理结果(dead->1)：",
              set(xls['patient.follow_ups.follow_up.vital_status']), end="->")
    xls = xls.dropna(subset=["patient.follow_ups.follow_up.vital_status"])
    if verbose:
        print(set(xls['patient.follow_ups.follow_up.vital_status']), end="")

  
    xls['patient.follow_ups.follow_up.vital_status'] = np.int0(
        xls['patient.follow_ups.follow_up.vital_status'] != "alive")
    
    if verbose:
        print("->", set(xls['patient.follow_ups.follow_up.vital_status']))
    if verbose:
        print("删除生存情况为nan的行后，剩余", xls.shape[0], "行数据\n")

  
    if verbose:
        print("存活天数<=0的行数：", xls[xls['months'] <= 0].shape[0])
    xls = xls[xls['months'] > 0]
    if verbose:
        print("删除存活天数<=0的行后，剩余", xls.shape[0], "行数据\n")
    if verbose:
        print("===================================")
    return xls


def getPerson(personFolder, filterFeatures=None) -> list:
   
    def getFeature(cellFolder, label, filter=None):
       
        pos = []
        feas = []
        labels = []
        for cell in os.listdir(cellFolder):
            xxx = cell.split(".png")[0].split("_")
            no, x, y = [int(i) for i in cell.split(".png")[0].split("_")]
            if (filter is not None) and (filter(no, x, y) == False): continue
           
            fea = np.load(os.path.join(cellFolder, cell))
            pos.append([x, y])
            feas.append(fea)
            labels.append(label)
        return np.array(pos), np.array(feas), np.array(labels)

    l_pos, l_fea, l_labels = getFeature(os.path.join(personFolder, "l"), 0, filterFeatures)  
    s_pos, s_fea, s_labels = getFeature(os.path.join(personFolder, "s"), 2, filterFeatures)  
    t_pos, t_fea, t_labels = getFeature(os.path.join(personFolder, "t"), 1, filterFeatures)  
   
    return [
        np.concatenate((l_pos, t_pos)),  
        np.concatenate((l_fea, t_fea)),  
        np.concatenate((l_labels, t_labels))  
    ]


def createGraphByKnn(pos, feats, k, ndata=None):
   

    

    def knn(pos, k=20):
        

        def calc_dist(x0, y0, x1, y1):
            
            return (x0 - x1) ** 2 + (y0 - y1) ** 2

        if pos.shape[0] < k:
            print("此图的顶点数(", pos.shape, ") < k(", k, "),无法选取最近的k个点组成图")
            raise Exception("此图的顶点数(", pos.shape, ") < k(", k, "),无法选取最近的k个点组成图")

        Dist = np.zeros((pos.shape[0], pos.shape[0]))
        Weights = []
        Strength = []

       
        for i in range(pos.shape[0]):
            j = i
            while j < pos.shape[0]:
                Dist[i][j] = Dist[j][i] = calc_dist(pos[i][0], pos[i][1], pos[j][0], pos[j][1])
                j += 1

        edges = []
        maxk = []
        for i in range(pos.shape[0]):
            maxKArgs = np.argpartition(Dist[i], k + 1)[:k + 1]
            maxk.append(maxKArgs)
            for j in maxKArgs:
               
                edges.append([i, j])
                if ndata is not None:
                   
                    Weights.append((2 ** ndata[i]) + (2 ** ndata[j]))
       
        maxk = np.array(maxk)
        for i in range(pos.shape[0]):
            for j in range(0, k + 1):
                temp = maxk[i, j]
                if i == temp:
                    Strength.append(6)
                    continue
                edge_value = 2 ** ndata[i] + 2 ** ndata[temp]
                if edge_value == 3: 
                    if i in maxk[temp]:
                        Strength.append(3)
                    else:
                        Strength.append(2)
                elif edge_value == 2: 
                    
                    if i in maxk[temp]:
                        Strength.append(1)
                    else:
                        Strength.append(0)
                else:
                    if i in maxk[temp]:  
                        Strength.append(5)
                    else:
                        Strength.append(4)

      
        return np.array(edges), np.array(Weights), np.array(Strength)

   
    def generateNet(edges, weights, strength, feats):
      
        _edges = torch.from_numpy(edges[:, 0]), torch.from_numpy(edges[:, 1])
       
        g = dgl.graph(_edges, idtype=torch.int32)
     
        g.edata['tag'] = torch.tensor(weights, dtype=torch.int32)
        g.edata['strength'] = torch.tensor(strength, dtype=torch.int32)
      
        g.ndata['feature'] = torch.tensor(feats, dtype=torch.float32)
       
        return g

    e, w, s = knn(pos, k)
    return generateNet(e, w, s, feats)


class TCGA_dataset(Dataset):
    def __init__(self,
                 folderPath: str = rp(r"J:/TMI_Dataset_BRCA"),
                 folderList: list = None,
                 excel_url=rp(r"BRCA/survival.xlsx"),
                 mRNA_url=rp(r"BRCA/RNAseq.xlsx"),
                 miRNA_url=rp(r"BRCA/miRNAseq.xlsx"),
                 k_nebs: int = 20,
                 max_handle_num: int = None,
                 filterFeatures: Callable = None,
                 verbose: bool = True,
                 transform_feats: Callable = None
                 ) -> None:
      
        super().__init__()
       
        self.tag = [] 
        self.lbl = [] 
        self.survtime = [] 
        self.k_nebs = 0  
        self.g = []  
        self.mRNA = []
        self.miRNA = []
        xls = handleXls(excel_url, verbose=verbose)
        mRNAxls =  pd.read_excel(mRNA_url)
        mRNAxls = mRNAxls.set_index(keys='patient.bcr_patient_barcode')
        miRNAxls = pd.read_excel(miRNA_url)
        miRNAxls = miRNAxls.set_index(keys='patient.bcr_patient_barcode')
        ls = []
        if folderList == None:
            ls = [os.path.join(folderPath, i) for i in sorted(
                os.listdir(folderPath))][:max_handle_num]
        else:
            ls = folderList[:max_handle_num]
       
        with tqdm(total=len(ls), desc="处理进度") as pbar:
            for personFolder in ls:
                tag = "-".join(os.path.split(personFolder)[-1].split("-")[0:3])
                pbar.set_description(tag)
                match_xls_res = xls[xls['patient.bcr_patient_barcode'] == tag.upper()].to_numpy()
                try:
                    mRNAseq = mRNAxls.loc[tag.upper()].to_numpy()
                    if len(mRNAseq.shape)>1:
                        mRNAseq = mRNAseq[0,:]
                    miRNAseq =  miRNAxls.loc[tag.upper()].to_numpy()
                    if len(miRNAseq.shape)>1:
                        miRNAseq = mRNAseq[0,:]
                    if match_xls_res.shape[0] == 0:
                       
                        pbar.update(1)
                        continue
                  
                    elif match_xls_res.shape[0] >= 2:
                        pass
                   
                    match_xls_res = match_xls_res[0]
                    tag, lbl, survtime = match_xls_res[0:3]  
                 
                    pos, feats, labs = getPerson(personFolder, filterFeatures)
                    if transform_feats != None:
                        feats = transform_feats(feats)
                        if type(feats) == type(None):
                         
                            pbar.update(1)
                            continue
                   
                    if pos.shape[0] < k_nebs:
                        pbar.update(1)
                        continue
                   
                    graph = createGraphByKnn(pos, feats, k=20, ndata=labs)
                    self.tag.append(tag)
                    self.lbl.append(lbl)
                    self.survtime.append(survtime)
                    self.g.append(graph)
                    self.mRNA.append(mRNAseq)
                    self.miRNA.append(miRNAseq)
                    pbar.update(1)
                except:
                    print('缺少基因数据')
            if verbose:
                print("read ", len(self.tag), " data")

    def __getitem__(self, index):
        return self.tag[index],\
            self.lbl[index],\
            self.survtime[index],\
            self.g[index],\
            self.mRNA[index],\
            self.miRNA[index],

    def __len__(self):
        return len(self.tag)


def getFolderList(
    folderPath: str = r"./features",
    train_size: float = 0.8,
    nums: int = -1,
    shuffle: bool = True,
    random_seed: int = 0,
    excludeList: list = []
) -> Tuple[list, list, list]:

    ls = sorted(
        filter(
            lambda x: x not in excludeList,
            os.listdir(folderPath)
        )
    )
    ls = [os.path.join(folderPath, i) for i in ls]
    if shuffle: 
        random.seed(random_seed)
        random.shuffle(ls)
  
    if nums != -1:
        ls = ls[:nums]
  
    nums = len(ls) if nums == -1 else nums
    train_num = int(nums*train_size)
    train_set = ls[: train_num]
    test_set = ls[train_num:]
    return ls, train_set, test_set


def load_model(model_path: str) -> Tuple[Dataset, torch.nn.Module]:
    
    f = open(model_path, "rb")
    model = pickle.loads(f.read())
    f.close()
    return model


def save_model(save_path: str, model: Tuple[Dataset, torch.nn.Module]) -> None:
   
    f = open(save_path, "wb")
    f.write(pickle.dumps(model))
    f.close()



brca_data = TCGA_dataset(
    folderList= None,
    excel_url=rp(r"BRCA/survival.xlsx"),
    mRNA_url = rp(r"BRCA/RNAseq.xlsx"),
    miRNA_url = rp(r"BRCA/miRNAseq.xlsx"),
    k_nebs=20,
    max_handle_num=None,
    verbose=False,
    transform_feats = lambda feats:PCA(n_components=64,random_state=0).fit_transform(feats) if min(feats.shape)>64 else None
)

save_model('BRCA/BRCA_dataset.pickle', brca_data)
pass
