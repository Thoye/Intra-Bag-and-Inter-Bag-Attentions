import numpy as np
import pickle
import codecs


class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos,sentlens):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos  # pos: position
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos
        self.sentlens = sentlens


def readData(filename, mode):
    print(filename)
    f = codecs.open(filename, 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        if mode == 1:
            num = line.split("\t")[3].strip().split(",")  # 把line这个数组进行分割,并取出分割后的第几个元素 num = 包中某个句子的id ???
        else:
            num = line.split("\t")[2].strip().split(",")
        ldist = []  # ldist = 头实体在句中的位置
        rdist = []  # rdist = 尾实体在句中的位置
        sentences = []
        sentlens = []
        entitiesPos = []
        pos = []
        rels = []
        for i in range(0, len(num)):  # len(num) = 字符串长度 ??
            sent = f.readline().strip().split(',')
            entities = sent[:2]  # entities = sent[0,1] = [实体1,实体2]
            epos = list(map(int,sent[2:4]))  # epos = sent[2,3] = [id1,id2]  List集合中Map对象,键是int类型,值是sent;
            epos = sorted(epos)
            rels.append(int(sent[4]))  # rels: relations = label (包的关系)  ???除了0还有别的数字???

            sent = f.readline().strip().split(",")  # 以id形式的句子
            sentences.append([(x+1) for x in list(map(int, sent))])
            sentlens.append(len(sent))  # 逐个增加词id的句子

            sent = f.readline().strip().split(",")
            ldist.append(list(map(int, sent)))  # ldist:当前词相对于头实体的位置

            sent = f.readline().strip().split(",")
            rdist.append(list(map(int, sent)))
            entitiesPos.append(epos)  # entitiesPos = [[],[],..]
            pos.append([0]*len(sentences[-1]))
        rels = list(set(rels))  # ???为啥除了0还有别的数字???
        # DocumentContainer()
        ins = DocumentContainer(entity_pair=entities, sentences=sentences, label=rels, pos=pos, l_dist=ldist, r_dist=rdist, entity_pos=entitiesPos, sentlens=sentlens)
        data += [ins]
    f.close()
    return data

def wv2pickle(filename='wv.txt', dim=50, outfile='Wv.p'):
    f = codecs.open(filename, 'r', encoding="utf-8")  # ??
    allLines = f.readlines()
    f.close()
    Wv = np.zeros((len(allLines)+1, dim))
    i = 1
    for line in allLines:
        line = line.split("\t")[1].strip()[:-1]
        Wv[i, :] = list(map(float, line.split(',')))
        i += 1
    rng = np.random.RandomState(3435)
    Wv[1, :] = rng.uniform(low=-0.5, high=0.5, size=(1, dim))
    f = codecs.open(outfile, 'wb')
    pickle.dump(Wv, f, -1)
    f.close()

def data2pickle(input, output, mode):
    data = readData(input, mode)
    f = open(output, 'wb')
    pickle.dump(data, f, -1)
    f.close()


if __name__ == "__main__":
    wv2pickle('word2vec.txt', 50, 'word2vec.pkl')  # ??
    data2pickle('bags_train.txt','train_temp.pkl',1)  # mode = 1? bags_train.txt在哪里
    data2pickle('bags_test.txt','test_temp.pkl',0)

