import itertools
import numpy as np
from torch.utils.data import Dataset as tDataset
import datetime
import os
import re
import pandas as pd
import requests
import torch

Debug = False
def debug(*args):
    if Debug:
        print(*args)

PAD_ID = 0
class DateData(tDataset):
    def __init__(self,n):
        np.random.seed(1)
        self.date_cn = []
        self.date_en = []
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        self.vocab= set(
            [str(i) for i in range(0,10)] + ["-","/","<GO>","<EOS>"] + [i.split("/")[1] for i in self.date_en]
        )
        self.v2i = {v:i for i,v in enumerate(sorted(list(self.vocab)), start=1)}
        self.v2i["<PAD>"] = PAD_ID
        self.vocab.add("<PAD>")
        self.i2v = {i:v for v,i in self.v2i.items()}
        self.x,self.y=[],[]
        for cn,en in zip(self.date_cn,self.date_en):
            self.x.append([self.v2i[v] for v in cn])
            self.y.append([self.v2i["<GO>"], ] + [self.v2i[v] for v in en[:3]] + [
                self.v2i[en[3:6]]] + [self.v2i[v] for v in en[6:]] + [self.v2i["<EOS>"],])
        self.x,self.y = np.array(self.x),np.array(self.y)
        self.start_token = self.v2i["<GO>"]
        self.end_token = self.v2i["<EOS>"]
    
    def __len__(self):
        return len(self.x)
    
    @property
    def num_word(self):
        return len(self.vocab)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index], len(self.y[index])-1
    
    def idx2str(self,idx):
        x=[]
        for i in idx:
            x.append(self.i2v[i])
            if i == self.end_token:
                break
        return "".join(x)

def pad_zero(seqs, max_len):
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded

class Dataset:
    def __init__(self,x,y,v2i,i2v):
        self.x,self.y = x,y
        self.v2i, self.i2v = v2i,i2v
        self.vocab = v2i.keys()
    
    def sample(self,n):
        b_idx = np.random.randint(0,len(self.x),n)
        bx,by = self.x[b_idx],self.y[b_idx]
        return bx,by
    @property
    def num_word(self):
        return len(self.v2i)

def process_w2v_data(corpus,skip_window=2,method = "skip_gram"):
    all_words = [sentence.split(" ") for sentence in corpus]
    # groups all the iterables together and produces a single iterable as output
    all_words = np.array(list(itertools.chain(*all_words)))
    vocab,v_count = np.unique(all_words,return_counts=True)
    vocab = vocab[np.argsort(v_count)[::-1]]
    
    print("All vocabularies are sorted by frequency in decresing oreder")
    v2i = {v:i for i,v in enumerate(vocab)}
    i2v = {i:v for v,i in v2i.items()}

    pairs = []
    js = [i for i in range(-skip_window,skip_window+1) if i!=0]

    for c in corpus:
        words = c.split(" ")
        w_idx = [v2i[w] for w in words]
        if method == "skip_gram":
            for i in range(len(w_idx)):
                for j in js:
                    if i+j<0 or i+j>= len(w_idx):
                        continue
                    pairs.append((w_idx[i],w_idx[i+j]))
        elif method.lower() == "cbow":
            for i in range(skip_window,len(w_idx)-skip_window):
                context = []
                for j in js:
                    context.append(w_idx[i+j])
                pairs.append(context+[w_idx[i]])
        else:
            raise ValueError
    
    pairs = np.array(pairs)
    print("5 expample pairs:\n",pairs[:5])
    if method.lower()=="skip_gram":
        x,y = pairs[:,0],pairs[:,1]
    elif method.lower() == "cbow":
        x,y = pairs[:,:-1],pairs[:,-1]
    else:
        raise ValueError
    return Dataset(x,y,v2i,i2v)

def maybe_download_mrpc(save_dir="./MRPC/", proxy=None):
    train_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_train.txt'
    test_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_test.txt'
    os.makedirs(save_dir, exist_ok=True)
    proxies = {"http": proxy, "https": proxy}
    for url in [train_url, test_url]:
        raw_path = os.path.join(save_dir, url.split("/")[-1])
        if not os.path.isfile(raw_path):
            print("downloading from %s" % url)
            r = requests.get(url, proxies=proxies)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(r.text.replace('"', "<QUOTE>"))
                print("completed")


def _text_standardize(text):
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
    text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
    return text.strip()


def _process_mrpc(dir="./MRPC", rows=None):
    data = {"train": None, "test": None}
    files = os.listdir(dir)
    for f in files:
        df = pd.read_csv(os.path.join(dir, f), sep='\t', nrows=rows)
        k = "train" if "train" in f else "test"
        data[k] = {"is_same": df.iloc[:, 0].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
    vocab = set()
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            for i in range(len(data[n][m])):
                data[n][m][i] = _text_standardize(data[n][m][i].lower())
                cs = data[n][m][i].split(" ")
                vocab.update(set(cs))
    v2i = {v: i for i, v in enumerate(sorted(vocab), start=1)}
    v2i["<PAD>"] = PAD_ID
    v2i["<MASK>"] = len(v2i)
    v2i["<SEP>"] = len(v2i)
    v2i["<GO>"] = len(v2i)
    i2v = {i: v for v, i in v2i.items()}
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            data[n][m+"id"] = [[v2i[v] for v in c.split(" ")] for c in data[n][m]]
    return data, v2i, i2v

class MRPCData(tDataset):
    num_seg = 3
    pad_id = PAD_ID

    def __init__(self, data_dir="./MRPC/", rows=None, data_type="train", proxy=None):
        """
        data_type: 'train' 或 'test', 用来区分加载训练集还是测试集
        """
        maybe_download_mrpc(save_dir=data_dir, proxy=proxy)
        data, self.v2i, self.i2v = _process_mrpc(data_dir, rows)
        
        # 根据 data_type 加载训练集或测试集
        self.raw_data = data[data_type]  # 注意这里的修改
        self.max_len = max(
            [len(s1) + len(s2) + 3 for s1, s2 in zip(
                data["train"]["s1id"] + data["test"]["s1id"], data["train"]["s2id"] + data["test"]["s2id"])]) # s1+s2+2*<SEP>+<GO>，所有句子中最长的1个
        
        self.xlen = np.array([
            [len(data[data_type]["s1id"][i]), len(data[data_type]["s2id"][i])]  # 这里的修改
            for i in range(len(data[data_type]["s1id"]))], dtype=int)
        
        x = [
            [self.v2i["<GO>"]] + data[data_type]["s1id"][i] + [self.v2i["<SEP>"]] + data[data_type]["s2id"][i] + [self.v2i["<SEP>"]]
            for i in range(len(self.xlen))
        ]  # <GO>+s1+<SEP>+s2+<SEP>
        
        self.x = pad_zero(x, max_len=self.max_len)  # padding 0
        
        self.nsp_y = data[data_type]["is_same"][:, None]

        self.seg = np.full(self.x.shape, self.num_seg-1, np.int32)
        for i in range(len(x)):
            si = self.xlen[i][0] + 2
            self.seg[i, :si] = 0
            si_ = si + self.xlen[i][1] + 1
            self.seg[i, si:si_] = 1
        # self.seg 第一句话0，第一句话1，padding补2
        self.word_ids = np.array(list(set(self.i2v.keys()).difference(
            [self.v2i[v] for v in ["<PAD>", "<MASK>", "<SEP>"]])))

    def __getitem__(self, idx):
        return self.x[idx], self.seg[idx], self.xlen[idx], self.nsp_y[idx]

    def sample(self, n):
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx, bs, bl, by = self.x[bi], self.seg[bi], self.xlen[bi], self.nsp_y[bi]
        return bx, bs, bl, by

    @property
    def num_word(self):
        return len(self.v2i)
    
    def __len__(self):
        return len(self.x)

    @property
    def mask_id(self):
        return self.v2i["<MASK>"]


class MRPCSingle(tDataset):
    pad_id = PAD_ID

    def __init__(self,data_dir="./MRPC/",rows = None, proxy= None):
        maybe_download_mrpc(save_dir=data_dir, proxy=proxy)

        data, self.v2i, self.i2v = _process_mrpc(data_dir, rows)

        self.max_len = max([len(s) + 2 for s in data["train"]["s1id"] + data["train"]["s2id"]])
        x = [
            [self.v2i["<GO>"]] + data["train"]["s1id"][i] + [self.v2i["<SEP>"]]
            for i in range(len(data["train"]["s1id"]))
        ]
        x += [
            [self.v2i["<GO>"]] + data["train"]["s2id"][i] + [self.v2i["<SEP>"]]
            for i in range(len(data["train"]["s2id"]))
        ]
        self.x = pad_zero(x, max_len=self.max_len)
        self.word_ids = np.array(list(set(self.i2v.keys()).difference([self.v2i["<PAD>"]])))
    def sample(self, n):
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx = self.x[bi]
        return bx

    @property
    def num_word(self):
        return len(self.v2i)
    
    def __getitem__(self, index):
        return self.x[index]

    
    def __len__(self):
        return len(self.x)
