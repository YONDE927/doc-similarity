# coding: UTF-8

import torch.nn as nn
import torch
import pandas as pd
from transformers import BertModel, BertJapaneseTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import pickle
import os

class Parser:
    """
    意味類似用の検索器ベースオブジェクト
    """
    def __init__(self):
        self.matrix=[]
        self.docs=[]
    def register(self,docs :list):
        pass
    def similardoc(self,text :str):
        pass
    def save(self,path="./data/matrix.pkl"):
        """
        登録文書と特徴量を保存

        Parameters
        ----------
        path : str
            保存先
        """
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"wb") as f:
            pickle.dump((self.docs,self.matrix),f)
    def load(self,path="./data/matrix.pkl"):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"rb") as f:
            self.docs,self.matrix = pickle.load(path)
class EnBertEmb(Parser):
        """
        登録文書と特徴量を読み込み

        Parameters
        ----------
        path : str
            保存先
        """
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.model= SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
        else:
            self.model= SentenceTransformer('all-MiniLM-L6-v2')
    def register(self,docs):
        """
        検索器にドキュメントを登録し、エンコード＆保持

        Parameters
        ----------
        docs : list<str>
            英語の文字列リスト
        Returns
        -------
        output : list
            n個のリストに対して、n * vocab_size の配列
        """
        self.matrix = []
        self.docs = docs
        for d in tqdm(docs):
            output = torch.Tensor(self.model.encode([d]))
            self.matrix.extend(output.to('cpu').tolist())
        return self.matrix
    def batch_register(self,docs,b=15):
        """
        検索器にドキュメントを登録し、エンコード＆保持
        バッチ処理で高速化できる。

        Parameters
        ----------
        docs : list<str>
            英語の文字列リスト
        b    : int
            バッチサイズ
        Returns
        -------
        output : list
            n個のリストに対して、n * vocab_size の配列
        """
        self.matrix = []
        self.docs = docs
        size=len(docs) // b
        for i in tqdm(range(size+1)):
            if i == size:
                batch = docs[i*b:]
            else:
                batch = docs[i*b:i*b+b]
            output = torch.Tensor(self.model.encode(batch))
            self.matrix.extend(output.to('cpu').tolist())
    def encode(self,endoc):
        return torch.Tensor(self.model.encode([endoc]))
    def similardoc(self,doc,size=-1):
        """
        登録されているコーパスから類似度の高い文章を降順で返す。

        Parameters
        ----------
        doc : str
            英語の文字列
        size : int
            返す文字列リストの要素数。-1だとすべて返す。
        Returns
        -------
        output : ndarray
            n個のリストに対して、n * vocab_size の行列
        """
        start = time.time()
        vector = self.encode(doc)
        corpas = torch.Tensor(self.matrix)
        if(len(self.matrix)>0):
            similarity = cosine_similarity(vector,corpas)
            ranking = similarity.squeeze().argsort()[::-1]
            print("%f sec for search"%(time.time()-start))
            if(size<0):
                size = len(self.docs)
            return [self.docs[i] for i in ranking[:size]]
        else:
            print("No docs are registerd.")

class JaBertEmb(Parser):
    def __init__(self, model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens-v2", device=None):
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def register(self, sentences, batch_size=1):
        """
        検索器にドキュメントを登録し、エンコード＆保持

        Parameters
        ----------
        docs : list<str>
            日本語の文字列リスト
        batch_size : int
            バッチサイズ
        Returns
        -------
        output : torhc.tensor
            n個のリストに対して、n * vocab_size の行列
        """
        self.docs = sentences
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in tqdm(iterator):
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)
        
        self.matrix = torch.stack(all_embeddings)
        # return torch.stack(all_embeddings).numpy()
        return self.matrix
    
    @torch.no_grad()
    def encode(self, sentences, batch_size=1):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in tqdm(iterator):
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)
        # return torch.stack(all_embeddings).numpy()
        return  torch.stack(all_embeddings)
    
    def similardoc(self,doc,size=-1):
        """
        登録されているコーパスから類似度の高い文章を降順で返す。

        Parameters
        ----------
        doc : str
            日本語の文字列
        size : int
            返す文字列リストの要素数。-1だとすべて返す。
        Returns
        -------
        output : ndarray
            n個のリストに対して、n * vocab_size の行列
        """
        start = time.time()
        vector = self.encode([doc])
        corpas = self.matrix
        if(len(self.matrix)>0):
            similarity = cosine_similarity(vector,corpas)
            ranking = similarity.squeeze().argsort()[::-1]
            print("%f sec for search"%(time.time()-start))
            if(size<0):
                size = len(self.docs)
            return [self.docs[i] for i in ranking[:size]]
        else:
            print("No docs are registerd.")
    


