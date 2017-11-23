#!/usr/bin/python3
# 
# get_trans.py
# Get transition sequence of transition-based parsers.
# Copyright 2017 Mengxiao Lin <linmx0130@gmail.com>
#

DEBUG_MODE = False
from ud_dataloader import UDToken, UDSentence

def no_depend_on(head_id, seq):
    for t in seq:
        if t.head == head_id:
            return False
    return True

def cross_check(tokens):
    seg = []
    for t in tokens:
        if t.head is None:
            continue
        a = t.wid
        b = t.head
        a, b = min(a,b), max(a,b)
        for s in seg:
            if a>s[0] and a < s[1] and b>s[1]:
                return False
            if b>s[0] and b < s[1] and a<s[0]:
                return False
        seg.append((a,b))
    return True

def get_transition_sequence(sentence: UDSentence):
    buf = sentence.tokens
    if not cross_check(buf):
        raise RuntimeError("Cross check failed!")

    seq = ['SHIFT', 'SHIFT', 'SHIFT']
    stack = buf[:3]
    buffer_idx = 3
    while buffer_idx < len(buf):
        if DEBUG_MODE:
            print("***")
            print("Seq = {}".format(seq))
            print("Stack = {}".format(stack))
        if len(stack)<=2:
            stack.append(buf[buffer_idx])
            buffer_idx = buffer_idx + 1
            seq.append('SHIFT')
            continue

        s1 = stack[-2]
        s2 = stack[-1]
        if s1.head==s2.wid: # left arc 
            stack.pop()
            stack.pop()
            stack.append(s2)
            seq.append('LEFT-ARC')
        elif s2.head==s1.wid and no_depend_on(s2.wid, buf[buffer_idx:]): # right arc
            stack.pop()
            stack.pop()
            stack.append(s1)
            seq.append('RIGHT-ARC')
        else: # shift
            stack.append(buf[buffer_idx])
            buffer_idx = buffer_idx + 1
            seq.append('SHIFT')

    while len(stack) > 1:
        if DEBUG_MODE:
            print("***")
            print("Seq = {}".format(seq))
            print("Stack = {}".format(stack))
        s1 = stack[-2]
        s2 = stack[-1]
        if s1.head==s2.wid: # left arc 
            stack.pop()
            stack.pop()
            stack.append(s2)
            seq.append('LEFT-ARC')
        elif s2.head==s1.wid and no_depend_on(s2.wid, buf[buffer_idx:]): # right arc
            stack.pop()
            stack.pop()
            stack.append(s1)
            seq.append('RIGHT-ARC')
        else:
            raise RuntimeError('Failed parsing!')
    return seq

