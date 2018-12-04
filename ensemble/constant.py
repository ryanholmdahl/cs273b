import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
if 'linux' in sys.platform:
    DATA_DIR = os.path.abspath('/data/data')
else:
    DATA_DIR = os.path.join(ROOT_PATH, 'data')
if 'linux' in sys.platform:
    SAVE_DIR = os.path.abspath('/data/save')
else:
    SAVE_DIR = os.path.join(ROOT_PATH, 'save')
GLOVE_DIR = os.path.join(DATA_DIR, 'glove')

TRAIN_IDS = os.path.join('/home/ryanlh', 'train_dbids_munoz.txt')
TEST_IDS = os.path.join('/home/ryanlh', 'test_dbids_munoz.txt')

TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels_1.pkl')
TEST_LABELS = os.path.join(DATA_DIR, 'test_labels_1.pkl')

TRAIN_LABEL_MATRIX = os.path.join(DATA_DIR, '../munoz2017_sider4_data/original/train_sideEffect')
TEST_LABEL_MATRIX = os.path.join(DATA_DIR, '../munoz2017_sider4_data/original/test_sideEffect')

TEXT_FEATURE_TAGS = ['description', 'indication', 'mechanism-of-action']