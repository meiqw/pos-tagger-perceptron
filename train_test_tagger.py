import sys
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence


def read_in_gold_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents 


def read_in_plain_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents 


def output_auto_data(filename, auto_data):
    ''' According to the data structure you used for "auto_data",
        write code here to output your auto tagged data into a file,
        using the same format as the provided gold data (i.e. word_pos word_pos ...). 
    '''
    with open(filename, 'w') as f:
        for sent in auto_data:
            tagged_sent = ""
            for word_tag in sent:
                tagged_sent += word_tag[0] + "_" + word_tag[1] + " "
            tagged_sent = tagged_sent.strip() + "\n"
            f.write(tagged_sent)



def get_tags(train_file):
    tags = []

    with open(train_file) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]

    for line in lines:
        for transition in line:
            tag = transition[1]
            if tag not in tags:
                tags.append(transition[1])

    return tags

if __name__ == '__main__':

    # Run python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt to train & test your tagger
    '''
    train_file = sys.argv[1]
    gold_dev_file = sys.argv[2]
    plain_dev_file = sys.argv[3]
    test_file = sys.argv[4]
    '''

    train_file = 'train/ptb_02-21.tagged'
    gold_dev_file = 'dev/ptb_22.tagged'
    plain_dev_file = 'dev/ptb_22.snt'
    test_file = 'test/ptb_23.snt'

    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)

    #get tags
    tags = get_tags(train_file)
    #print(tags)

    # Train your tagger
    my_tagger = Perceptron_POS_Tagger(tags)
    weight, ave_weight = my_tagger.train(train_data, gold_dev_data, 1)


    # Apply your tagger on dev & test data
    #using regular perceptron
    auto_dev_data = []
    auto_test_data = []
    for sent in plain_dev_data:
        auto_dev_data.append(my_tagger.tag(sent, weight))
    for sent in test_data:
        auto_test_data.append(my_tagger.tag(sent, weight))

    # Outpur your auto tagged data
    output_auto_data("dev_pos.txt", auto_dev_data)
    output_auto_data("test_pos.txt", auto_test_data)
