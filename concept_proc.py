import os
import pickle
import torch
import json
from tqdm import tqdm
from models.CAB.others_layer import glove_embedding
from utils.common_layer import TransEncoder
from utils.config import args
from utils import config
from data.process.util import wordCate, Stack, lemmatize_all
import csv
from textrank4zh import TextRank4Keyword
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from ast import literal_eval

REMOVE_RELATIONS = ["ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "SymbolOf", "CreatedBy", "MadeOf"]

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# print(arg.multi_hop)
def get_concept_dict():
    '''
    Retrieve concepts from ConceptNet using the EmpatheticDialogue tokens as queries
    :return:
    '''

    if(os.path.exists('../data/dataset_preproc.p')):
        print("LOADING ed data for get concept dict")
        with open('../data/dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
            word2index = vocab.word2index

    embeddings = glove_embedding(vocab)
    CN = csv.reader(open("../data/assertions.csv", "r", encoding="utf-8"))

    concept_dict = {}
    concept_file = open("../data/ConceptNet.json", "w", encoding="utf-8")

    relation_dict = {}
    rd = open("../data/relation.json", "w", encoding="utf-8")

    for i, row in enumerate(CN):
        if i%1000000 == 0:
            print("Processed {} rows".format(i))
        #按照tab对原始的行进行分割，共有5个元素
        items = "".join(row).split("\t")
        #第二个元素的第二个斜杠之后是语言类型
        language = items[2].split("/")[2]
        if language == "en":
            if len(items) != 5:
                print("concept error")
            #第一个元素的第二个斜杠之后是关系
            relation = items[1].split("/")[2]
            c1 = items[2].split("/")[3]
            c2 = items[3].split("/")[3]
            #进行词形还原
            c1 = wnl.lemmatize(c1)
            c2 = wnl.lemmatize(c2)
            #置信度分数
            weight = literal_eval("{" + row[-1].strip())["weight"]
            score = (weight - 1) / (10 - 1)  # 最大最小归一化
            if weight < 1.0:  # filter tuples where confidence score is smaller than 1.0
                continue
            if c1 in word2index and c2 in word2index and c1 != c2 and c1.isalpha() and c2.isalpha():#检查一个字符串中所有的字符是否都是由字母构成的，并且至少有1个字符
                if relation not in word2index:
                    if relation in relation_dict:
                        relation_dict[relation] += 1
                    else:
                        relation_dict[relation] = 0
                #shape:300 一行向量
                c1_vector = torch.Tensor(embeddings(torch.tensor([[word2index[c1]]]))[0][0])
                c2_vector = torch.Tensor(embeddings(torch.tensor([[word2index[c2]]]))[0][0])
                c1_c2_sim = torch.cosine_similarity(c1_vector, c2_vector, dim=0).item()
                # <c1 relation c2>
                if c2 not in stop_words:
                    if c1 in concept_dict:
                        concept_dict[c1][c2] = [relation, c1_c2_sim, score]
                    else:
                        concept_dict[c1] = {}
                        concept_dict[c1][c2] = [relation, c1_c2_sim, score]
                # reverse relation  <c2 relation c1>
                if c1 not in stop_words:
                    if c2 in concept_dict:
                        concept_dict[c2][c1] = ['Reverse-' +relation, c1_c2_sim, score]
                    else:
                        concept_dict[c2] = {}
                        concept_dict[c2][c1] = ['Reverse-' +relation, c1_c2_sim, score]
    print("concept num: ", len(concept_dict))
    json.dump(concept_dict, concept_file)

    relation_dict = sorted(relation_dict.items(), key=lambda x: x[1], reverse=True)
    json.dump(relation_dict, rd)

    rank_concept_file = open('../data/ConceptNet_ranked_dict.json', 'w', encoding='utf-8')
    rank_concept = {}
    for i in concept_dict:
        # 根据(score)由大到小排序
        rank_concept[i] = dict(sorted(concept_dict[i].items(), key= lambda x: x[1][2], reverse= True))
        # 将字典转换成列表：{'':[[头实体,r,cos,s],[],……],' ':[[],[]]……}
        rank_concept[i] = [[l, rank_concept[i][l][0], rank_concept[i][l][1], rank_concept[i][l][2]] for l in rank_concept[i]]
    json.dump(rank_concept, rank_concept_file, indent=4)

def get_encode(token, sentence, vocab,embeddings,encoder):
    word2index = vocab.word2index
    sentence = torch.tensor(sentence).unsqueeze(0)
    mask = sentence.data.eq(config.PAD_idx).unsqueeze(1)

    token_emb = embeddings(torch.tensor([[word2index[token]]]))
    sentence_emb = embeddings(sentence)

    token_hidden = encoder(sentence_emb, token_emb, mask)

    return token_hidden[0][0]

def extract_triples(data, vocab):
    concept = json.load(open("../knowledge_data/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    embeddings = glove_embedding(vocab)
    encoder = TransEncoder(args.emb_dim, args.hidden_dim, num_layers=1, num_heads=args.heads,
                           total_key_depth=args.depth, total_value_depth=args.depth, filter_size=args.filter,
                           universal=args.universal)
    word2index = vocab.word2index

    for q, three_tri in tqdm(enumerate(data['triples']), total= len(data['triples'])):
        if three_tri == [] and len(data['key_concepts'][q]) > 1:
            context = data['context_s'][q]
            context_id = []
            for i, sentence in enumerate(context):
                context_id += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                       sentence]  #得到句子的id表示
            if config.USE_CUDA:
                context_id = torch.tensor(context_id).cuda()

            #构建关键概念之间的候选实体
            G = [] #上下文概念集合
            # flag = False #判断当前是否已经找到了路径
            J = Stack() #上下文概念堆栈
            key_concepts = [] #当前对话进过筛选的关键词
            for word in data['key_concepts'][q]:
                if word in word2index:
                    key_concepts.append(word)
            for con in key_concepts:
                tail = set(key_concepts) - {con}
                tail = list(tail)

                for ta in tail:
                    h = 0 #记录多跳数
                    J.push((' ', ' ', con))
                    g = []
                    while not J.is_empty():
                        head = J.pop(J.peek())
                        h +=1
                        g.append(head)

                        # 该关键概念的所有候选实体
                        candi_c = [concept[head[2]]]
                        #筛选出K个与头实体相似的实体
                        triples = []  # 三元组列表
                        # head_vector = embeddings(torch.tensor([[word2index[head[2]]]]).cuda())[0][0]
                        head_vector = get_encode(head[2], context_id, vocab, embeddings,encoder)
                        if candi_c != []:#如果当前这个关键概念有实体
                            for x, c in enumerate(candi_c[0]):
                                if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                    candi_vector = embeddings(torch.tensor([[word2index[c[0]]]]))[0][0]
                                    head_cos_candi = torch.cosine_similarity(head_vector, candi_vector, dim=0).item()
                                    socre = head_cos_candi + c[3]
                                    triples.append([head[2], c[1], c[0], socre])
                        triples = sorted(triples, key=lambda x: x[3], reverse=True)  # 按照与头实体的相似度从大到小进行排序
                        triples = triples[:args.K_num]
                        # 筛选出k个与尾实体相关的实体
                        # ta_vector = embeddings(torch.tensor([[word2index[ta]]]).cuda())[0][0]
                        ta_vector = get_encode(ta, context_id, vocab, embeddings, encoder)
                        for tr in triples:
                            tr_vector = embeddings(torch.tensor([[word2index[tr[2]]]]))[0][0]
                            ta_cos_tr = torch.cosine_similarity(ta_vector, tr_vector, dim=0).item()
                            tr.append(ta_cos_tr)
                        triples = sorted(triples, key= lambda x: x[4], reverse= True)#按照与尾实体的相似度从大到小进行排序
                        triples = triples[: args.k_num]
                        triples = triples[::-1]

                        for tr in triples:
                            if tr[2] == ta:
                                g.append((tr[0],tr[1],tr[2]))
                                G.append(g)
                                J.clean()
                                break
                            else:
                                if h < args.multi_hop:#当超过了规定的多跳数且尾实体不满足条件时就不将其添加到J中
                                    J.push((tr[0],tr[1],tr[2]))
                        # 如果达到了当前的最大跳数还没有找到符合的路径，则退出当前的路径，更换并行的另一条路径
                        if h == args.multi_hop:
                            h -= 1
                            g = g[:-1]

                            if not J.is_empty() and J.peek()[0] != head[0]:
                                g = g[:-1]
            G = [n for n in G if len(n) > 2]
            # G = sorted(G, key= lambda x: len(x), reverse= True)
            # G = G[: args.path_num]
            data['triples'][q] = G
    return data

def re_extract_triples(data, vocab):
    concept = json.load(open("../knowledge_data/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    embeddings = glove_embedding(vocab)
    encoder = TransEncoder(args.emb_dim, args.hidden_dim, num_layers=1, num_heads=args.heads,
                           total_key_depth=args.depth, total_value_depth=args.depth, filter_size=args.filter,
                           universal=args.universal)
    word2index = vocab.word2index

    for q, three_tri in tqdm(enumerate(data['triples']), total= len(data['triples'])):
        if three_tri == []:
            context = data['context_s'][q]
            context_id = []
            for i, sentence in enumerate(context):
                context_id += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                       sentence]  #得到句子的id表示
            if config.USE_CUDA:
                context_id = torch.tensor(context_id).cuda()

            #构建关键概念之间的候选实体
            G = [] #上下文概念集合
            # flag = False #判断当前是否已经找到了路径
            J = Stack() #上下文概念堆栈
            key_concepts = [] #当前对话进过筛选的关键词
            for word in data['key_concepts'][q]:
                if word in word2index:
                    key_concepts.append(word)
            for con in key_concepts:
                h = 0 #记录多跳数
                J.push((' ', ' ', con))
                g = []
                select_num = False
                while not J.is_empty():
                    head = J.pop(J.peek())
                    g.append(head)
                    # 该关键概念的所有候选实体
                    candi_c = [concept[head[2]]]
                    #筛选出K个与头实体相似的实体
                    triples = []  # 三元组列表
                    # head_vector = embeddings(torch.tensor([[word2index[head[2]]]]).cuda())[0][0]
                    head_vector = get_encode(head[2], context_id, vocab, embeddings,encoder)
                    if candi_c != []:#如果当前这个关键概念有实体
                        for x, c in enumerate(candi_c[0]):
                            if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                candi_vector = embeddings(torch.tensor([[word2index[c[0]]]]))[0][0]
                                head_cos_candi = torch.cosine_similarity(head_vector, candi_vector, dim=0).item()
                                socre = abs(head_cos_candi) + c[3]
                                triples.append([head[2], c[1], c[0], socre])
                    triples = sorted(triples, key=lambda x: x[3], reverse=True)  # 按照与头实体的相似度从大到小进行排序
                    if triples == []:
                        G.append(g)
                        break
                    if triples[0][2] == g[-1][0]: #判断当前节点的尾结点是否与上一个节点的头结点一样，避免重复选择
                        select_num = True
                    else:
                        select_num = False
                    if select_num:
                        triples = triples[1:args.K_num+1]
                    else:
                        triples = triples[:args.K_num]
                    for tr in triples:
                        h+=1
                        J.push((tr[0],tr[1],tr[2]))
                    if tr[2] == con or h > args.multi_hop:
                        g.append((tr[0],tr[1],tr[2]))
                        G.append(g)
                        J.clean()
            G = [n for n in G if len(n) > 2]
            # G = sorted(G, key= lambda x: len(x), reverse= True)
            # G = G[: args.path_num]
            data['triples'][q] = G
    return data

def key_extract_triples(data, vocab):
    concept = json.load(open("../knowledge_data/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    embeddings = glove_embedding(vocab)
    encoder = TransEncoder(args.emb_dim, args.hidden_dim, num_layers=1, num_heads=args.heads,
                           total_key_depth=args.depth, total_value_depth=args.depth, filter_size=args.filter,
                           universal=args.universal)
    word2index = vocab.word2index

    for q, three_tri in tqdm(enumerate(data['triples']), total= len(data['triples'])):
        concepts = {}
        if three_tri == []:
            context = data['context_s'][q]
            for i in context:
                for j in i:
                    concepts[j] = word2index[j]
            concepts = dict(sorted(concepts.items(), key=lambda x: x[1], reverse=False))
            words_pos = nltk.pos_tag(concepts.keys())
            dialog_concepts = [
                word if word not in stop_words and wordCate(words_pos[wi]) else ''
                for wi, word in enumerate(concepts)]

            context_id = []
            for i, sentence in enumerate(context):
                context_id += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                       sentence]  #得到句子的id表示
            if config.USE_CUDA:
                context_id = torch.tensor(context_id).cuda()

            #构建关键概念之间的候选实体
            G = [] #上下文概念集合
            # flag = False #判断当前是否已经找到了路径
            J = Stack() #上下文概念堆栈
            key_concepts = [] #当前对话进过筛选的关键词
            for word in dialog_concepts:
                if word in word2index and word in concept:
                    key_concepts.append(word)
            for con in key_concepts:
                h = 0 #记录多跳数
                J.push((' ', ' ', con))
                g = []
                select_num = False
                while not J.is_empty():
                    head = J.pop(J.peek())
                    g.append(head)
                    # 该关键概念的所有候选实体
                    candi_c = [concept[head[2]]]
                    #筛选出K个与头实体相似的实体
                    triples = []  # 三元组列表
                    # head_vector = embeddings(torch.tensor([[word2index[head[2]]]]).cuda())[0][0]
                    head_vector = get_encode(head[2], context_id, vocab, embeddings,encoder)
                    if candi_c != []:#如果当前这个关键概念有实体
                        for x, c in enumerate(candi_c[0]):
                            if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                candi_vector = embeddings(torch.tensor([[word2index[c[0]]]]))[0][0]
                                head_cos_candi = torch.cosine_similarity(head_vector, candi_vector, dim=0).item()
                                socre = abs(head_cos_candi) + c[3]
                                triples.append([head[2], c[1], c[0], socre])
                    triples = sorted(triples, key=lambda x: x[3], reverse=True)  # 按照与头实体的相似度从大到小进行排序
                    if triples == []:
                        G.append(g)
                        break
                    if triples[0][2] == g[-1][0]: #判断当前节点的尾结点是否与上一个节点的头结点一样，避免重复选择
                        select_num = True
                    else:
                        select_num = False
                    if select_num:
                        triples = triples[1:args.K_num+1]
                    else:
                        triples = triples[:args.K_num]
                    if triples == []:
                        G.append(g)
                        break
                    for tr in triples:
                        h+=1
                        J.push((tr[0],tr[1],tr[2]))
                    if tr[2] == con or h > args.multi_hop:
                        g.append((tr[0],tr[1],tr[2]))
                        G.append(g)
                        J.clean()
            G = [n for n in G if len(n) > 1]
            # G = sorted(G, key= lambda x: len(x), reverse= True)
            # G = G[: args.path_num]
            data['triples'][q] = G
    return data

def get_sample_data(data):
    for key, value in data.items():
            data[key] = value[:10000]
    return data

def create_context_concept():
    '''
    给定一个对话的emotion_s_txt
    :return:该txt的概念以及选取的三元组
    '''
    # print(os.getcwd())
    if(os.path.exists('../ed_concept_dataset_preproc.p')):
        print("LOADING ed data for create context concept")
        with open('../ed_concept_dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    with open('../prepare_data/all/valid_dataset_preproc.json', "r") as f:
        data_val = json.load(f)

    # data_tra = get_sample_data(data_tra)
    # data_val = get_sample_data(data_val, train = False)
    # data_tst = get_sample_data(data_tst, train = False)

    # tra = key_extract_triples(data_tra, vocab)
    val = key_extract_triples(data_val, vocab)
    # tst = key_extract_triples(data_tst, vocab)

    fp = open('../prepare_data/all/valid_dataset_preproc.json', "w")
    json.dump(val, fp)
    print("Saved JSON")




if __name__ == '__main__':

    # get_concept_dict()

    create_context_concept()

