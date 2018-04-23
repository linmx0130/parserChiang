import sys
sys.path.append("..")

train_data_fn = 'data/CTB5/ctb_train.conllu'
dev_data_fn = 'data/CTB5/ctb_test.conllu'
prompt_inteval = 50
PARSER_TAGS = ['SHIFT', 'LEFT-ARC', 'RIGHT-ARC']
PARSER_TAGS_MAP = {'SHIFT':0 , 'LEFT-ARC': 1, 'RIGHT-ARC': 2}
UNKNOW_TOKEN = '[UNK]'
UPDATE_STEP = 10
EMBED_SIZE = 100
NUM_HIDDEN = 125
POS_EMBED_SIZE = 25
PUNC_POS_TAG = 'PU'
