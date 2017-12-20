#!/usr/bin/python3
# 
# ud_dataloader.py
# Data loader of Universal Dependencies dataset.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#
import hashlib

class UDToken:
    wid = None
    form = None
    lemma = None
    pos_tag = None
    x_pos_tag = None
    feats = None
    head = -1
    deprel = None
    deps = None
    misc = None
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
        items = input_line.strip().split()
        assert len(items)==10
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
    def __init__(self, lines):
        self.parseLines(lines)
    
    def parseLines(self, lines):
        lines = [t for t in lines if not t.startswith('#')]
        # self.sent_id = lines[0].split('=')[1].strip()
        # self.text = lines[1].split('=')[1].strip()
        self.tokens = [UDToken()]
        try:
            for t in lines:
                self.tokens.append(UDToken(t))
        except Exception as e:
            print("Parsing error at {}: {}".format(self.text, str(e)))
            self.error_found = True
            return
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
            if t.startswith('# newdoc id'):
                continue
            if len(t)!=0:
                sentence_pd.append(t)
            else:
                data_by_sentences.append(sentence_pd)
                sentence_pd = []
    if len(sentence_pd) > 0:
        data_by_sentences.append(ret)
    ret = []
    for sentence_chunk in data_by_sentences:
        ret.append(UDSentence(sentence_chunk))
    ret = list(filter(lambda x:not x.error_found, ret))
    return ret

def mask_pos_with_x(sen: UDSentence):
    for x in sen.tokens:
        x.pos_tag = x.x_pos_tag

