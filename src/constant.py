import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
if 'linux' in sys.platform:
    DATA_DIR = os.path.abspath('/data/data')
else:
    DATA_DIR = os.path.join(ROOT_PATH, 'data')
GLOVE_DIR = os.path.join(DATA_DIR, 'glove')

TRAIN_IDS = os.path.join(DATA_DIR, 'train_dbids.txt')
TEST_IDS = os.path.join(DATA_DIR, 'test_dbids.txt')
TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels_1.pkl')
TEST_LABELS = os.path.join(DATA_DIR, 'test_labels_1.pkl')

feature_tags = ['description', 'indication', 'mechanism-of-action']