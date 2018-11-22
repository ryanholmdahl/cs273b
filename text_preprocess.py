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
        #print(dbids)
        return dbids


def read_text_features(doc, db_id_list):
    length = len(doc['drugbank']['drug'])

    dicts = {tag: {} for tag in feature_tags}

    find_id = []
    miss_id = []
    for dbid in db_id_list:
        no_id = True
        for num in range(length):
            if type(doc['drugbank']['drug'][num]['drugbank-id']) == list:
                xml_dbid = [doc['drugbank']['drug'][num]['drugbank-id'][0]['#text']] \
                           + [doc['drugbank']['drug'][num]['drugbank-id'][i] for i in
                              range(1, len(doc['drugbank']['drug'][num]['drugbank-id']))]
            else:
                xml_dbid = [doc['drugbank']['drug'][num]['drugbank-id']['#text']]
            if dbid in xml_dbid:
                no_id = False
                find_id.append(dbid)
                for tag in feature_tags:
                    if tag in doc['drugbank']['drug'][num]:
                        dicts[tag][dbid] = doc['drugbank']['drug'][num][tag]
                    else:
                        dicts[tag][dbid] = NO_VALUE
        if no_id:
            miss_id.append(dbid)
    print('Missing drug id:', miss_id)
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
    with open(os.path.join(constant.DATA_DIR,'drugbank.xml')) as fd:
        doc = xmltodict.parse(fd.read())
    train_feature_dicts = read_text_features(doc, train_db_id)
    test_feature_dicts = read_text_features(doc, test_db_id)
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
word_vocab.add_medw2v_to_vocab(constant.DATA_DIR)
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

# print('Getting embedding vectors')
# with open(os.path.join(constant.DATA_DIR,'embed_vectors.pkl'), 'wb') as fd:
#     pickle.dump(word_vocab.get_glove_embed_vectors(), fd)

missing_cnt = 0
for i, token in word_vocab.index2word.items():
    if i < 3:  # Skip the first 3 words PAD EOS UNK
        continue
    # token = token.strip(''',.:;"()'/?<>[]{}\|!@#$%^&*''')
    # print(i,token,(token in stoi))
    if token in word_vocab.glove_stoi:
        pass
    else:
        try:
            word_vocab.medw2v_model[token]
        except KeyError:
            print(i, token.encode('utf-8'))
            missing_cnt += 1
        # print(i, token)
print('Miss', missing_cnt, 'words')

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

