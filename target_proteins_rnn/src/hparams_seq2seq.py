import os
import sys

HIDDEN_SIZE = 20 # hidden_size is the feature vector dimension we eventually have
AGGREG_METHOD = 'MEAN'
LEARNING_RATE = 1e-3
N_EPOCHS = 2
TEACHER_ENFORCING = True

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 5039 # 5038 + one EOS

ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
if 'linux' in sys.platform:
    DATA_DIR = os.path.abspath('/data/data_targetProteins')
else:
    DATA_DIR = os.path.join(ROOT_PATH, 'data')
if 'linux' in sys.platform:
    SAVE_DIR = os.path.abspath('/data/save_targetProteins')
else:
    SAVE_DIR = os.path.join(ROOT_PATH, 'output')

LETTER_TO_TEXT_FILE = os.path.join(DATA_DIR, 'train_seq_letter_vocab.pkl')
TRAIN_SEQ_FILE = os.path.join(DATA_DIR, 'train_target_seq.pkl')
SAVE_PATH = os.path.join(SAVE_DIR, 
                         'seq2seq_hiddenDim{}_teacherForcing{}_'.format(HIDDEN_SIZE, TEACHER_ENFORCING))
