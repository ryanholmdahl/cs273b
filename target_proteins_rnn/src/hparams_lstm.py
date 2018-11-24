import os

HIDDEN_SIZE = 20 # hidden_size is the feature vector dimension we eventually have
EMBEDDING_DIM = 32
AGGREG_METHOD = 'MEAN'
LEARNING_RATE = 1e-3
N_EPOCHS = 100

PAD_token = 0 # unused
SOS_token = 1
EOS_token = 2
UNK_token = 3


ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(ROOT_PATH, 'data')
GOWORDS_VOCAB_FILE = os.path.join(DATA_DIR, 'gowords_to_idx.pkl')
GOTERMS_VOCAB_FILE = os.path.join(DATA_DIR, 'goterms_to_idx.pkl')
TRAIN_GOTERMS_FILE = os.path.join(DATA_DIR, 'train_target_go.pkl')
GOTERMS_LABEL_FILE = os.path.join(DATA_DIR, 'train_go_indices.pkl')
SAVE_PATH_GO = os.path.join(ROOT_PATH, 
                         'output/goterms_hiddenDim{}_embeddingDim{}_'.format(HIDDEN_SIZE, EMBEDDING_DIM))

TARGET_LETTER_VOCAB_FILE = os.path.join(DATA_DIR, 'target_letter_vocab.pkl')
TARGET_VOCAB_FILE = os.path.join(DATA_DIR, 'target_vocab.pkl')
TRAIN_TARGET_FILE = os.path.join(DATA_DIR, 'train_target_IDs.pkl')
TARGET_LABEL_FILE = os.path.join(DATA_DIR, 'train_target_indices.pkl')
SAVE_PATH_TARGET = os.path.join(ROOT_PATH, 
                                'output/target_hiddenDim{}_embeddingDim{}_'.format(HIDDEN_SIZE, EMBEDDING_DIM))
