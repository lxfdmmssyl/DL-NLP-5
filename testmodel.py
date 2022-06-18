import torch
import pickle



with open('./sentences.pickle', 'rb') as f:
    datas = pickle.load(f)

from gensim.models import word2vec
model_cbow = word2vec.Word2Vec.load('./cbow.model')
mark = ['。”','？”','！”','。','？','！']

encoder = torch.load('./model/encoder20000')
dncoder = torch.load('./model/decoder20000')
print('read model over')

if __name__ == "__main__":
    index = 3
    sen = 100
    for i in range(sen, sen+20):
        print(datas[index][sen])
