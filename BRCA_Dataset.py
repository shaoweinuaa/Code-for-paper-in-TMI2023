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
    """获取相对于此文件的地址

    Args:
        path (str): 路径

    Returns:
        str: 真实文件路径
    """
    try:
        return os.path.join(os.path.dirname(__file__), path)
    except NameError:
        return path


def handleXls(excel_url: str, verbose=True) -> pd.DataFrame:
    """处理表格数据

    Args:
        excel_url (str): 表格文件地址
        verbose (bool, optional): 是否输出log. Defaults to True.

    Returns:
        pd.DataFrame: 处理好的表格
    """
    if verbose:
        print("===================================")
    # %% 处理表格
    xls = pd.read_excel(excel_url)
    if verbose:
        print("读入", xls.shape[0], "行数据\n")
    # 把表格中的病人编号全部转为大写
    xls['patient.bcr_patient_barcode'] = xls['patient.bcr_patient_barcode'].str.upper()

    # 删除 生存情况 为 nan 的行
    if verbose:
        print("生存情况的处理结果(dead->1)：",
              set(xls['patient.follow_ups.follow_up.vital_status']), end="->")
    xls = xls.dropna(subset=["patient.follow_ups.follow_up.vital_status"])
    if verbose:
        print(set(xls['patient.follow_ups.follow_up.vital_status']), end="")

    # 把生存情况alive、dead分别转为数字0、1
    xls['patient.follow_ups.follow_up.vital_status'] = np.int0(
        xls['patient.follow_ups.follow_up.vital_status'] != "alive")
    #xls['patient.follow_ups.follow_up.vital_status']=[1 if random.random()>0.5 else 0 for i in range(len(xls['patient.follow_ups.follow_up.vital_status']))]
    if verbose:
        print("->", set(xls['patient.follow_ups.follow_up.vital_status']))
    if verbose:
        print("删除生存情况为nan的行后，剩余", xls.shape[0], "行数据\n")

    # 删除存活天数小于0的行
    if verbose:
        print("存活天数<=0的行数：", xls[xls['months'] <= 0].shape[0])
    xls = xls[xls['months'] > 0]
    if verbose:
        print("删除存活天数<=0的行后，剩余", xls.shape[0], "行数据\n")
    if verbose:
        print("===================================")
    return xls


def getPerson(personFolder, filterFeatures=None) -> list:
    # 返回  pos=[ [x,y], ... ],
    #      features=[ [...], ... ],
    #      labels=[ type, ... ] (type in [l,s,t])
    def getFeature(cellFolder, label, filter=None):
        # cellFolder : 细胞文件夹，如 .../l/ ,.../s/
        # filter=[ None | lambda no,x,y:Boolean ] 筛选加入列表的patch
        pos = []
        feas = []
        labels = []
        for cell in os.listdir(cellFolder):
            xxx = cell.split(".png")[0].split("_")
            no, x, y = [int(i) for i in cell.split(".png")[0].split("_")]
            # 若开启了筛选函数并且筛选结果为False，则跳过此patch
            if (filter is not None) and (filter(no, x, y) == False): continue
            # fea为单个细胞的特征，shape=[1,1000]->fea=fea[0]
            # eg.  cellFolder:'E:/fixed/features/features/TCGA-E2-A1LA-01Z-00-DX1\\l'

            # fea = np.load(os.path.join(cellFolder, cell))["feature"][0]
            fea = np.load(os.path.join(cellFolder, cell))
            pos.append([x, y])
            feas.append(fea)
            labels.append(label)
        return np.array(pos), np.array(feas), np.array(labels)

    l_pos, l_fea, l_labels = getFeature(os.path.join(personFolder, "l"), 0, filterFeatures)  # 淋巴细胞 300
    s_pos, s_fea, s_labels = getFeature(os.path.join(personFolder, "s"), 2, filterFeatures)  # 基质细胞 300
    t_pos, t_fea, t_labels = getFeature(os.path.join(personFolder, "t"), 1, filterFeatures)  # 肿瘤细胞 300
    # return [
    #         np.concatenate((l_pos,s_pos,t_pos)),# 位置信息
    #         np.concatenate((l_fea,s_fea,t_fea)),# 特征信息
    #         np.concatenate((l_labels,s_labels,t_labels)) # 标签信息
    # ]
    return [
        np.concatenate((l_pos, t_pos)),  # 位置信息
        np.concatenate((l_fea, t_fea)),  # 特征信息
        np.concatenate((l_labels, t_labels))  # 标签信息
    ]


def createGraphByKnn(pos, feats, k, ndata=None):
    """

    Args:
        pos: [[x,y]] 坐标 的np array
        feats:图的节点特征
        k: knn default=50
        ndata:

    Returns:

    """

    def knn(pos, k=20):
        # ndata是顶点的类别信息（基质、肿瘤、淋巴），
        # 会根据ndata对建立的图的边进行标注（6种）

        # 输入形如 pos = [ [x,y] ,..] 的np array
        # 计算任意两个点间的欧氏距离，得到距离矩阵Dist
        # Dist[i][j]表示i与j的距离

        # 对Dist的每一行i，获取第k大的数nk（快排O(nlgn)）
        # 若Dist[i][j]<=nk,边设为1，否则设为0

        def calc_dist(x0, y0, x1, y1):
            # 计算xy欧式距离的平方
            # return (x-y)^2
            return (x0 - x1) ** 2 + (y0 - y1) ** 2

        if pos.shape[0] < k:
            print("此图的顶点数(", pos.shape, ") < k(", k, "),无法选取最近的k个点组成图")
            raise Exception("此图的顶点数(", pos.shape, ") < k(", k, "),无法选取最近的k个点组成图")

        # 初始Dist全为0
        Dist = np.zeros((pos.shape[0], pos.shape[0]))
        Weights = []
        Strength = []

        # 更新距离矩阵Dist
        for i in range(pos.shape[0]):
            j = i
            while j < pos.shape[0]:
                Dist[i][j] = Dist[j][i] = calc_dist(pos[i][0], pos[i][1], pos[j][0], pos[j][1])
                j += 1

        # 对每行Dist[i]选出前k小的Dist[i][j]，创建ij之间的边
        # 边集edges=[ [i,j], ... ]
        edges = []
        maxk = []
        # TODO 这里默认每个细胞与自己是建立连接的
        for i in range(pos.shape[0]):
            maxKArgs = np.argpartition(Dist[i], k + 1)[:k + 1]
            maxk.append(maxKArgs)
            for j in maxKArgs:
                # if i==j:continue
                edges.append([i, j])
                if ndata is not None:
                    # # 设立边的值，表示边是由哪两个类型的点建立的，共6种：00 11 22 01 02 12 -> (2**i)+(2**j) ->  2 4 8 3 5 6
                    # Weights.append( (2**ndata[i]) + (2**ndata[j]) )
                    # 设立边的属性值，表示边是由哪两个类型的点建立的，共3种：00 01 11 -> (2**i)+(2**j) ->  双淋巴 2  淋肿 3  双肿瘤 4
                    Weights.append((2 ** ndata[i]) + (2 ** ndata[j]))
        # 设置边的强度值，表示01边的相近程度，共6 + 1种：
        # 淋淋单：0 淋淋双 ：1 淋肿单 ： 2 淋肿双 ：3 肿肿单：4 肿肿双:5  自环：6
        maxk = np.array(maxk)
        for i in range(pos.shape[0]):
            # 遍历每个i行的topk
            for j in range(0, k + 1):
                # temp为i行j个对应的index：说明i，temp之间有边[i,temp]
                temp = maxk[i, j]
                # 如果temp是本身 continue
                if i == temp:
                    Strength.append(6)
                    continue
                # todo 确认边为01边
                edge_value = 2 ** ndata[i] + 2 ** ndata[temp]
                if edge_value == 3:  # 淋巴 肿瘤
                    # 判断[temp,i]之间有边
                    if i in maxk[temp]:
                        Strength.append(3)
                    else:
                        Strength.append(2)
                elif edge_value == 2:  # 淋巴
                    # 判断[temp,i]之间有边
                    if i in maxk[temp]:
                        Strength.append(1)
                    else:
                        Strength.append(0)
                else:
                    if i in maxk[temp]:  # 肿瘤
                        Strength.append(5)
                    else:
                        Strength.append(4)

        # print(len(edges)/pos.shape[0])
        # return Dist
        # todo 回传Strength
        return np.array(edges), np.array(Weights), np.array(Strength)

    # 根据knn生成的边关系建立dgl图
    def generateNet(edges, weights, strength, feats):
        # 输入形如np.array([[a,b],...])，表示构建的图是a-b的有向图
        _edges = torch.from_numpy(edges[:, 0]), torch.from_numpy(edges[:, 1])
        # print(_edges)
        # 构建有向图
        g = dgl.graph(_edges, idtype=torch.int32)
        # 赋予边信息（6种关系）
        g.edata['tag'] = torch.tensor(weights, dtype=torch.int32)
        g.edata['strength'] = torch.tensor(strength, dtype=torch.int32)
        # todo 添加节点属性
        g.ndata['feature'] = torch.tensor(feats, dtype=torch.float32)
        # 转为无向图 弃用
        # g=dgl.to_bidirected(g)
        """
            为什么使用有向图
            考虑一种情况：
                结点a的最近邻顺序是b,c,d,f,g
                结点b的最近邻顺序是c,d,e,f,a
            如果选最近的3个点建立近邻图，
            那么就有 a->b a->c a->d
                    b->c b->d b->e
            不难发现，a->b是一条边，但b->a并不在其中
        """
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
        """TCAG数据集

        Args:
            x (_type_): _description_
            folderPath (str, optional): sample父文件夹存放地址. Defaults to rp(r"../data/features").
            folderList (list, optional): 可传入若干sample文件夹地址的元组，当此项不为None时，folderPath无效.
            excel_url (str, optional): sample表格地址. Defaults to rp(r"../data/survival.xlsx").
            k_nebs (int, optional): 每个sample选取多少个近邻的patch进行构图. Defaults to 50.
            max_handle_num (int, optional): 最大读入病例数量，默认=None，即全部读入. Defaults to None.
            filterFeatures (Callable, optional): 对选取的特征进行筛选 lambda no,x,y:True|False. Defaults to None.
            verbose (bool, optional): 是否输出log. Defaults to True.
            transform_feats (Callable, optional): 输入一个函数变换每个patch的feats. 如果返回值为None，则忽视当前sample. Defaults to None.
        """
        super().__init__()
        # 初始化变量
        self.tag = []  # 病人标签
        self.lbl = []  # 是否存活
        self.survtime = []  # 存活时间
        self.k_nebs = 0  # 一个病人选取多少个近邻的细胞进行构图
        self.g = []  # 根据细胞位置构建的近邻图
        self.mRNA = []
        self.miRNA = []
        # 读取表格数据
        xls = handleXls(excel_url, verbose=verbose)
        mRNAxls =  pd.read_excel(mRNA_url)
        mRNAxls = mRNAxls.set_index(keys='patient.bcr_patient_barcode')
        miRNAxls = pd.read_excel(miRNA_url)
        miRNAxls = miRNAxls.set_index(keys='patient.bcr_patient_barcode')
        # 读入病人文件夹
        ls = []
        if folderList == None:
            ls = [os.path.join(folderPath, i) for i in sorted(
                os.listdir(folderPath))][:max_handle_num]
        else:
            ls = folderList[:max_handle_num]
        # 筛选符合要求的病人数据
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
                        # 若该病人不在表格中，则忽视此病人
                        pbar.update(1)
                        continue
                    # 存在多个病人对应表格中的一行数据，则按病人编号排序取第一个
                    elif match_xls_res.shape[0] >= 2:
                        pass
                    # 取编号靠前的第一个病人
                    match_xls_res = match_xls_res[0]
                    tag, lbl, survtime = match_xls_res[0:3]  # 分别为病人tag，是否存活，存活时间
                    # 读取sample的patch信息
                    pos, feats, labs = getPerson(personFolder, filterFeatures)
                    if transform_feats != None:
                        feats = transform_feats(feats)
                        if type(feats) == type(None):
                            # 若变换特征的函数存在且返回的值为None，则跳过
                            pbar.update(1)
                            continue
                    # 根据病人细胞位置构建图
                    if pos.shape[0] < k_nebs:
                        # 若病人的细胞块数量小于k_nebs，无法构图，则跳过
                        pbar.update(1)
                        continue
                    # graph = createGraphByKnn(
                    #     pos=pos,
                    #     k=k_nebs,
                    #     node_feas={
                    #         "lst": labs,
                    #         "feats": feats
                    #     }
                    # )
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
    """获取所有sample的路径并划分训练集、测试集

    Args:
        folderPath (str, optional): sample的父目录. Defaults to r"./features".
        train_size (float, optional): 训练集比例. Defaults to 0.8.
        nums (int, optional): 读取的sample数. Defaults to -1,表示全部读取.
        shuffle (bool, optional): 是否打乱集合. Defaults to True.
        random_seed (int, optional): 打乱集合的随机数种子. Defaults to 0.
        excludeList (list, optional): 排除的sample列表. Defaults to [].

    Returns:
        Tuple[list,list,list]: all samples、train samples、test samples
    """
    ls = sorted(
        filter(
            lambda x: x not in excludeList,
            os.listdir(folderPath)
        )
    )
    ls = [os.path.join(folderPath, i) for i in ls]
    if shuffle:  # 随机排序
        random.seed(random_seed)
        random.shuffle(ls)
    # 是否限制病人数量
    if nums != -1:
        ls = ls[:nums]
    # 划分训练集、测试集
    nums = len(ls) if nums == -1 else nums
    train_num = int(nums*train_size)
    train_set = ls[: train_num]
    test_set = ls[train_num:]
    return ls, train_set, test_set


def load_model(model_path: str) -> Tuple[Dataset, torch.nn.Module]:
    """读取处理好的保存在本地的数据集/模型

    Args:
        model_path (str): 数据集/模型的路径

    Returns:
        Tuple[Dataset,GatNet]: Dataset类型的数据集/nn.Module类型的模型
    """
    f = open(model_path, "rb")
    model = pickle.loads(f.read())
    f.close()
    return model


def save_model(save_path: str, model: Tuple[Dataset, torch.nn.Module]) -> None:
    """保存处理好的数据集/训练好的模型到本地

    Args:
        save_path (str): 保存路径
        dataset (Dataset): 待保存的数据集
    """
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
