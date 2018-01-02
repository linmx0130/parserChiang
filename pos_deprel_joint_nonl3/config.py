import sys
sys.path.append("..")

train_data_fn = 'data/en-ud-train.conllu'
dev_data_fn = 'data/en-ud-test.conllu'
prompt_inteval = 20
PARSER_TAGS = ['SHIFT', 'LEFT-ARC', 'RIGHT-ARC']
PARSER_TAGS_MAP = {'SHIFT':0 , 'LEFT-ARC': 1, 'RIGHT-ARC': 2}
UNKNOW_TOKEN = '[UNK]'
UPDATE_STEP = 4
NUM_EMBED = 100
NUM_HIDDEN = 125
TAG_EMBED = 25
PUNC_POS_TAG = 'PUNCT'
WORD_DROPOUT_RATE = 0.25
