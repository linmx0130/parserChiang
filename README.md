parserChiang - Naïve Transition-based Dependency Parser in Gluon
=====
This repo support CoNLL format, which is adapted by [Universal Dependencies](http://universaldependencies.org/) Project. parserChiang is implemented with great [MXNet gluon](http://gluon.mxnet.io/).

### Usage
Data should be put into *data/* directory. Train the model with
<pre>
$ python3 train_parser.py
</pre>
Then it will create a directory named *model_dumps_{Date}_{Time}* to store the model dump. Test it with
<pre>
$ python3 test_parser.py [model_path] [model_file]
</pre>

### POS Tag Joint Training Model
In *pos_joint/*, a parser model with POS tagging is also provided. It will utilize the information of POS tags. The usage of this model is similar to naive model.

### Notes
This implementation is a **low**-performance transition-based parser in both training speed and predicition accuracy. I created it as a toy model simply for learning natural language processing. **DO NOT USE IT IN ANY REAL WORLD TASKS**. 

Have fun with it!

### License
Copyright 2017 Mengxiao Lin \<linmx0130@gmail.com\>, read LICENSE for more details.
