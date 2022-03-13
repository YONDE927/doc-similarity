import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

class Mcparser:
    """
    mecabの出力に品詞フィルタを掛けるオブジェクト

    Attributes
    ----------
    self.mecab : Mecab
        mecab-python3のパーサー
    """
    def __init__(self):
        self.mecab = MeCab.Tagger("-r /home/ec2-user/SageMaker/mecab/mecab-0.996/mecabrc -d /usr/local/lib/mecab/dic/ipadic -O chasen")
        #self.mecab = MeCab.Tagger("-O chasen")
    def parse(self,text):
        res = self.mecab.parse(text)
        nodes=[]
        for line in res.split('\n'):
            tmp=line.split('\t')
            #print(tmp)
            if(len(tmp)==6):
                nodes.append(tmp)
        return nodes

    def extract(self,text,stop_keys):
        """
        文章から品詞に相当する形態素を基本形のリストで返す

        Parameters
        ----------
        text : str
            解析対象となる日本語の文章
        keys : list<str>
            品詞のリスト

        Returns
        -------
        output : list<str>
            フィルタ済みの基本形リスト
        """
        output=[]
        nodes = self.parse(text)
        #print(nodes)
        for n in nodes:
            flag=True
            for k in stop_keys:
                if(k in n[3]):
                    flag=False
            if flag:
                output.append(n[2])
        return ' '.join(output)

class Tfidf:
    """
    Tfidfに関するインターフェース
    - tfidfの構築
    - ドキュメントのベクトル化
    - 次元削減

    Attributes
    ----------
    self.mecab : Mecab
        mecab-python3のパーサー
    self.stop_words = []
    self.stop_pos = []
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.mcparser = Mcparser()
        self.matrix = np.array([])
        self.stop_words = []
        self.stop_pos = []
        
    def set_stop_word(self,words):
        self.stop_words = words
        
    def set_stop_pos(self,poses):
        self.stop_pos = poses
        
    def train(self,docs):
        """
        コーパスからTfidfを計算。パラメータ保持のみ

        Parameters
        ----------
        docs : list<str>
            日本語の文字列のリスト
        """
        dataset=[]
        for d in docs:
            output=self.mcparser.extract(d,self.stop_pos)
            #print(output)
            dataset.append(output)
        print(dataset)
        self.vectorizer.fit(dataset)
        
    def transform(self,docs):
        """
        計算済みのTfidfを用いて入力をベクトル化

        Parameters
        ----------
        docs : list<str>
            日本語の文字列のリスト
        Returns
        -------
        output : ndarray
            n個のリストに対して、n * vocab_size の行列
        """
        res=[self.mcparser.extract(d,self.stop_pos) for d in tqdm(docs)]
        return self.vectorizer.transform(res).toarray()
    
    def fit_transform(self,docs):
        """
        コーパスからTfidfを計算。そして入力をベクトル化
        また、matrixというインスタンス変数に行列を格納

        Parameters
        ----------
        docs : list<str>
            日本語の文字列のリスト
        Returns
        -------
        output : ndarray
            n個のリストに対して、n * vocab_size の行列
        """
        res=[self.mcparser.extract(d,self.stop_pos) for d in tqdm(docs)]
        self.matrix = self.vectorizer.fit_transform(res).toarray()
        return self.matrix
    
class TfidfSearchEngine(Tfidf):
    def __init__(self):
        super().__init__()
    def register(self,docs):
        self.docs = docs
        super().fit_transform(docs)
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
        vector = super().transform([doc])
        if(self.matrix.size>0):
            similarity = cosine_similarity(vector,self.matrix)
            ranking = similarity.squeeze().argsort()[::-1]
            print("%f sec for search"%(time.time()-start))
            if(size<0):
                size = len(self.docs)
            return [self.docs[i] for i in ranking[:size]]
        else:
            print("No docs are registerd.")
    