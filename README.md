parserChiang - Naïve Transition-based Dependency Parser in Gluon
=====
This repo support CoNLL format, which is adapted by [Universal Dependencies](http://universaldependencies.org/) Project. parserChiang is implemented with great [MXNet gluon](http://gluon.mxnet.io/).

### Models
There are different models in this repo:
1. [DEPRECATED] *default/*: The default parser model using only word features. It is the baseline of all other models.
2. [DEPRECATED] *pos_aid/*: This parser model requires standard POS tagging during inference, which is provided in CoNLL dataset. In practice, you may use Stanford NLP tools to get good POS tags.
3. [DEPRECATED] *pos_joint/*: This parser model will predict POS tags. 
4. *pos_deprel_joint/*: This parser model will predict POS tags and **dependent relation label**. LAS index requires the output from this model. 
5. *pos_aid_deprel_joint/*: This parser model requires standard POS tagging during inference, and will predict ependent relation label.

The models marked with [DEPRECATED] will not be updated to latest functions.

### Usage
Data should be put into *data/* directory. Train the model with
<pre>
$ python3 train_parser.py
</pre>
Then it will create a directory named *model_dumps_{Date}_{Time}* to store the model dump. Test it with
<pre>
$ python3 test_parser.py [model_path] [model_file]
</pre>

### Notes
This implementation is a **low**-performance transition-based parser in both training speed and predicition accuracy. I created it as a toy model simply for learning natural language processing. **DO NOT USE IT IN ANY REAL WORLD TASKS**. 

Have fun with it!

### License
Copyright 2017 Mengxiao Lin \<linmx0130@gmail.com\>, read LICENSE for more details.
