import os
import jieba
import re
import pickle

def load_data(path, ban_stop_words=False, stop_words_path='', add_words=False, add_words_path=''):
    max_len = 0
    if ban_stop_words:
        stop_words = set()
        stop_txt = os.listdir(stop_words_path)
        for file in stop_txt:
            with open(stop_words_path + '/' + file, 'r', encoding='ANSI') as f:
                for j in f.readlines():
                    stop_words.add(j.strip('\n'))

    replace = '[a-zA-Z0-9’"#$%&\'()（）*+-./「<=>@★【】《》[\\]^_`{|}~]+\n\u3000 \t'

    add_txt = os.listdir(add_words_path)

    if add_words:
        for file in add_txt:
            with open(add_words_path + '/' + file, 'r', encoding='ANSI') as f:
                for j in f.readlines():
                    jieba.add_word(j.strip('\n'))

    all = []
    files = os.listdir(path)
    for file in files:
        all_text = []
        with open(path + '/' + file, 'r', encoding='ANSI') as f:
            t = f.read()
            for i in replace:
                t = t.replace(i, '')

            t = re.split('(。”|？”|！”|。|？|！)', t)
            mark = ['。”','？”','！”','。','？','！']

            for i in range(len(t)-1):
                if t[i] not in mark:
                    if t[i+1] in mark:
                        all_text.append(jieba.lcut(t[i]) + [t[i+1]])
                    else:
                        all_text.append(jieba.lcut(t[i]))
        all.append(all_text)
        print("{} loaded".format(file))


    with open('./sentences/all.txt', 'w', encoding='utf-8') as f:
        for i in all:
            for j in i:
                if len(j) > max_len:
                    max_len = len(j)
                for k in j:
                    f.write(k)
                    f.write(' ')
                f.write('\n')

    with open('./sentences.pickle', 'wb') as f:
        pickle.dump(all, f)

    print(max_len)
    return


if __name__ == '__main__':
    ban_stop_words = False
    add_words = True
    load_data("./data", ban_stop_words, "./stop", add_words, './words')


