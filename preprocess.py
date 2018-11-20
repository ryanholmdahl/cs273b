# ID_IDX = 0
#
# # text
# DES_IDX = 4
# IND_IDX = 11
# ACT_IDX = 13
#
# #class
# CLS_IDX = 5

import xmltodict
import csv
import src.dataman.vocab as vocab
import pickle
import os
import src.constant as constant

NO_VALUE = ""
feature_tags = ['description', 'indication', 'mechanism-of-action']


def read_sider_db_map(filename):
    db_id = []
    sd_id = []
    with open(filename) as fd:
        reader = csv.reader(fd, delimiter=',')
        header = next(reader)
        for row in reader:
            db_id.append(row[0])
            sd_id.append(int(row[1]))
    return db_id, sd_id


def read_dbids_txt(filename):
    with open(filename, "r") as fd:
        lines = fd.readlines()
        dbids = [line.strip('\n') for line in lines]
        return dbids


def read_text_features(filename, db_id_list):
    with open(filename) as fd:
        doc = xmltodict.parse(fd.read())
    length = len(doc['drugbank']['drug'])

    dicts = {tag: {} for tag in feature_tags}

    find_id = []
    for dbid in db_id_list:
        for num in range(length):
            if type(doc['drugbank']['drug'][num]['drugbank-id']) == list:
                xml_dbid = doc['drugbank']['drug'][num]['drugbank-id'][0]['#text']
            else:
                xml_dbid = doc['drugbank']['drug'][num]['drugbank-id']['#text']
            if xml_dbid == dbid:
                find_id.append(dbid)
                for tag in feature_tags:
                    if tag in doc['drugbank']['drug'][num]:
                        dicts[tag][dbid] = doc['drugbank']['drug'][num][tag]
                    else:
                        dicts[tag][dbid] = NO_VALUE
    return dicts


# train_db_id, train_sd_id = read_sider_db_map('duplicates_removed_train.csv')
# test_db_id, test_sd_id = read_sider_db_map('duplicates_removed_test.csv')

train_db_id = read_dbids_txt(os.path.join(constant.DATA_DIR,"train_dbids.txt"))
test_db_id = read_dbids_txt(os.path.join(constant.DATA_DIR,"test_dbids.txt"))

print("train:",len(train_db_id),", test:", len(test_db_id))

train_cache_file_name = os.path.join(constant.DATA_DIR,"raw_train_features")
test_cache_file_name = os.path.join(constant.DATA_DIR,"raw_test_features")
if os.path.isfile(train_cache_file_name) and os.path.isfile(test_cache_file_name):
    print('Reading from cache file')
    with open(train_cache_file_name, 'rb') as cache:
        train_feature_dicts = pickle.load(cache)
    with open(test_cache_file_name, 'rb') as cache:
        test_feature_dicts = pickle.load(cache)
else:
    print('Reading from xml')
    train_feature_dicts = read_text_features(os.path.join(constant.DATA_DIR,'drugbank.xml'), train_db_id)
    test_feature_dicts = read_text_features(os.path.join(constant.DATA_DIR,'drugbank.xml'), test_db_id)
    with open(train_cache_file_name, 'wb') as cache:
        pickle.dump(train_feature_dicts, cache)
    with open(test_cache_file_name, 'wb') as cache:
        pickle.dump(test_feature_dicts, cache)

word_vocab = vocab.Vocab()

print('Parsing sentences')
for tag, feature_dict in train_feature_dicts.items():
    for dbid, sent in feature_dict.items():
        word_vocab.addSentence(sent)
        #print(sent)
print("Vocabulary size:",word_vocab.n_words)
word_vocab.add_glove_to_vocab(constant.GLOVE_DIR, 300)
print("Vocabulary size:",word_vocab.n_words)

print('Numberizing sentences')
maxlen = 0
len_bin = [0 for i in range(10)]

token_dicts = {tag: {} for tag in feature_tags}
for tag, feature_dict in train_feature_dicts.items():
    for dbid, sent in feature_dict.items():
        token_dicts[tag][dbid] = word_vocab.numberize_sentence(sent)
        if len(token_dicts[tag][dbid]) > maxlen:
            maxlen = len(token_dicts[tag][dbid])
        len_bin[int(len(token_dicts[tag][dbid])/100)] += 1
        #print(token_dicts[tag][dbid])

for tag, feature_dict in test_feature_dicts.items():
    for dbid, sent in feature_dict.items():
        token_dicts[tag][dbid] = word_vocab.numberize_sentence(sent)
        if len(token_dicts[tag][dbid]) > maxlen:
            maxlen = len(token_dicts[tag][dbid])
        len_bin[int(len(token_dicts[tag][dbid])/100)] += 1
        #print(token_dicts[tag][dbid])
print(maxlen)
print(len_bin)

print('Getting embedding vectors')
word_vocab.get_glove_embed_vectors()
with open(os.path.join(constant.DATA_DIR,'embed_vectors.pkl'), 'wb') as fd:
    pickle.dump(word_vocab.embed_vectors, fd)

print('Saving files')
# for tag in feature_tags:
#     with open(tag+'_raw.pkl', 'wb') as fd:
#         pickle.dump(train_feature_dicts[tag], fd)
for tag in feature_tags:
    with open(os.path.join(constant.DATA_DIR,'text_'+'dbid2'+tag+'_tok.pkl'), 'wb') as fd:
        pickle.dump(token_dicts[tag], fd)

with open(os.path.join(constant.DATA_DIR,'text_idx2word.pkl'), 'wb') as fd:
    pickle.dump(word_vocab.index2word, fd)
with open(os.path.join(constant.DATA_DIR,'text_word2idx.pkl'), 'wb') as fd:
    pickle.dump(word_vocab.word2index, fd)
with open(os.path.join(constant.DATA_DIR,'text_word2cnt.pkl'), 'wb') as fd:
    pickle.dump(word_vocab.word2count, fd)

