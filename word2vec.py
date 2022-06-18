from gensim.models import word2vec


if __name__ == '__main__':
    '''
    sentences = word2vec.LineSentence('./sentences/all.txt')
    
    model_cbow = word2vec.Word2Vec(sentences, sg=0, vector_size=256, window=5, min_count=0, workers=3, epochs=10)
    model_cbow.save('./cbow.model')
    '''
    model_cbow = word2vec.Word2Vec.load('./cbow.model')


    print(model_cbow.wv.most_similar(positive=model_cbow.wv['太后'], topn=1))
    print(model_cbow.wv.most_similar('杨过', topn=10))
    print(model_cbow.wv.most_similar('郭靖', topn=10))
    print(model_cbow.wv.most_similar('韦小宝', topn=10))
    print(model_cbow.wv.most_similar('峨嵋派', topn=10))
    print(model_cbow.wv.most_similar('六脉神剑', topn=10))
    print(model_cbow.wv.most_similar('。”', topn=10))





