#!/usr/bin/python3
# 
# ud_dataloader.py
# Data loader of Universal Dependencies dataset.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#
import hashlib

class UDToken:
    wid = "_"
    form = "_"
    lemma = "_"
    pos_tag = "_"
    x_pos_tag = "_"
    feats = "_"
    head = -1
    deprel = "_"
    deps = "_"
    misc = "_"
    def __init__(self, data_str=""):
        if len(data_str)==0:
            self.wid = 0
            self.form = '[ROOT]'
            self.lemma = '[ROOT]'
            self.pos_tag = '[ROOT]'
            self.x_pos_tag = '[ROOT]'
        else:
            self.parseLine(data_str)
    
    def parseLine(self, input_line):
        items = input_line.strip().split('\t')
        assert len(items)==10, print(input_line, '\n', items, len(items))
        try:
            self.wid = int(items[0])
            self.form = items[1]
            self.lemma = items[2]
            self.pos_tag = items[3]
            self.x_pos_tag = items[4]
            self.feats = items[5]
            self.head = int(items[6])
            self.deprel = items[7]
            self.deps = items[8]
            self.misc = items[9]
        except Exception as e:
            raise e

    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return '({}, {})'.format(self.wid, self.form)

class UDSentence:
    sent_id = None
    text = None
    error_found = True
    tokens = None
    def __init__(self, lines=None):
        if lines is not None:
            self.parseLines(lines)

    def parseLines(self, lines):
        lines = [t for t in lines if not t.startswith('#')]
        # self.sent_id = lines[0].split('=')[1].strip()
        # self.text = lines[1].split('=')[1].strip()
        tokens = [UDToken()]
        try:
            for t in lines:
                try:
                    tokens.append(UDToken(t))
                except ValueError:
                    pass
        except Exception as e:
            print("Parsing error at {}: {}".format(self.text, str(e)))
            self.error_found = True
            return
        self.set_token_list(tokens)

    def set_token_list(self, tokens):
        self.tokens = tokens
        self.error_found = False
        self.text = " ".join([t.form for t in self.tokens])
        self.sent_id = hashlib.sha1(self.text.encode()).hexdigest()

    def __len__(self):
        return len(self.tokens)
    def __str__(self):
        return repr(self) 
    
    def __repr__(self):
        return "{}: {}".format(self.sent_id, self.text)
        
def parseDocument(filename):
    with open(filename) as f:
        data = [t.strip() for t in f.readlines()]
        data_by_sentences = []
        sentence_pd = []
        for t in data:
            if t.startswith('#'):
                continue
            if len(t)!=0:
                sentence_pd.append(t)
            else:
                data_by_sentences.append(sentence_pd)
                sentence_pd = []
    if len(sentence_pd) > 0:
        data_by_sentences.append(sentence_pd)
    ret = []
    for sentence_chunk in data_by_sentences:
        ret.append(UDSentence(sentence_chunk))
    ret = list(filter(lambda x:not x.error_found, ret))
    return ret

def nltkParseDocument(filename):
    import nltk
    nltk.download('punkt')
    ret = []
    with open(filename) as f:
        data = [t.strip() for t in f.readlines()]
    for sen in data:
        tokens_str = nltk.word_tokenize(sen)
        ud_tokens = [UDToken()]
        for wid, item in enumerate(tokens_str):
            token = UDToken()
            token.wid = wid + 1
            token.form = item 
            token.lemma = item
            token.pos_tag = None
            token.x_pos_tag = None
            ud_tokens.append(token)
        ud_sen = UDSentence()
        ud_sen.set_token_list(ud_tokens)
        ret.append(ud_sen)
    return ret

def mask_pos_with_x(sen: UDSentence):
    for x in sen.tokens:
        x.pos_tag = x.x_pos_tag

def get_x_pos_of_punct(sen: UDSentence, punct_tag=None):
    ret = set()
    for x in sen.tokens:
        if x.pos_tag == punct_tag:
            ret.add(x.x_pos_tag)
    return ret


class UDWriter:
    def __init__(self, filename:str):
        self._handle = open(filename, 'w')
    def write_token(self, token: UDToken):
        output_line = "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n"%(
                token.wid, token.form, token.lemma, token.pos_tag,
                token.x_pos_tag, token.feats, token.head, 
                token.deprel, token.deps, token.misc)
        self._handle.write(output_line)

    def write_sentence(self, sentence: UDSentence):
        for token in sentence.tokens[1:]:
            self.write_token(token)
        self._handle.write("\n")

    def close(self):
        self._handle.close()
