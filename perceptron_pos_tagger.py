from sparse_vector import Vector
from collections import defaultdict
from data_structures import Sentence

class Perceptron_POS_Tagger(object):
    def __init__(self,tags):
        ''' Modify if necessary. 
        '''
        self.tags = tags

    def tag(self, test_sent, weight):
        ''' Implement the Viterbi decoding algorithm here.
        '''
        #print("viterbi")
        #print("weight: ", weight.v)

        path = []
        if len(test_sent.get_snt()) == 0:
            return path

        trellis = defaultdict(lambda: defaultdict(int))
        backpointer = {}

        # initialization
        backpointer[0] = {}
        word_1 = ""
        word_2 = "START"
        word1 = test_sent.get_snt()[1] if 1 < len(test_sent.get_snt()) else ""
        word2 = test_sent.get_snt()[2] if 2 < len(test_sent.get_snt()) else ""
        for tag in self.tags:
            trellis[0][tag] = weight.dot(test_sent.local_feature(test_sent.get_snt()[0], '<S>', tag, word_1, word_2, word1, word2))
            #print("START local feature: ", test_sent.local_feature(test_sent.get_snt()[0], '<S>', tag))
            #print("weight: ", weight)
            #print("initialization score: ", weight.dot(test_sent.local_feature(test_sent.get_snt()[0], '<S>', tag)))
            backpointer[0][tag] = '<S>'

        #recursive steps
        for i in range(1, len(test_sent.get_snt())):
            backpointer[i] = {}
            word_1 = test_sent.get_snt()[i - 2] if i - 2 > 0 else ""
            word_2 = test_sent.get_snt()[i - 1] if i - 1 > 0 else ""
            word1 = test_sent.get_snt()[i + 1] if i + 1 < len(test_sent.get_snt()) else ""
            word2 = test_sent.get_snt()[i + 2] if i + 2 < len(test_sent.get_snt()) else ""
            if i == 1:
                word_1 = "START"
            if i == len(test_sent.get_snt()) - 2:
                word2 = "$END"
            if i == len(test_sent.get_snt()) - 1:
                word1 = "$END"

            for cur_tag in self.tags:
                max_score = 0
                best_tag = self.tags[0]
                for prev_tag in self.tags:
                    cur_score = weight.dot(test_sent.local_feature(test_sent.get_snt()[i], prev_tag, cur_tag, word_1, word_2, word1, word2)) + \
                        trellis[i-1][prev_tag]
                    #print("local features: ", test_sent.local_feature(test_sent.get_snt()[i], prev_tag, cur_tag))
                    #print("weight: ", weight)
                    #print("recursive steps score: ", cur_score)
                    if cur_score > max_score:
                        max_score = cur_score
                        best_tag = prev_tag

                trellis[i][cur_tag] = max_score
                #print("max score each block: ", max_score)
                backpointer[i][cur_tag] = best_tag

        max_score = 0
        best_tag = self.tags[0]
        #termination steps
        backpointer[len(test_sent.get_snt())] = {}
        word_1 = test_sent.get_snt()[-2] if 2 < len(test_sent.get_snt()) else ""
        word_2 = test_sent.get_snt()[-1] if 1 < len(test_sent.get_snt()) else ""
        word1 = ""
        word2 = ""
        for tag in self.tags:
            cur_score = weight.dot(test_sent.local_feature('$END', tag, '</S>', word_1, word_2, word1, word2)) + \
                        trellis[len(test_sent.get_snt())-1][tag]
            #print("END local feature: ", test_sent.local_feature('$END', tag, '</S>'))
            #print("weight: ", weight)
            if cur_score > max_score:
                max_score = cur_score
                best_tag = tag
        trellis[len(test_sent.get_snt())]['</S>'] = max_score
        #print("final score: ", max_score)
        backpointer[len(test_sent.get_snt())]['</S>'] = best_tag

        #backtracing
        current_tag = best_tag
        t = len(test_sent.get_snt()) - 1
        path = [[test_sent.get_snt()[t], current_tag]]

        #current_tag = backpointer[t-1][current_tag]
        # path.append([test_sent[l-1], cur_tag])
        while t > 0:
            t -= 1
            current_tag = backpointer[t+1][current_tag]
            path.insert(0, [test_sent.get_snt()[t], current_tag])

        return path

    def train(self, train_data, dev_data, iterations):
        ''' Implement the Perceptron training algorithm here.
        '''
        weight = Vector()
        #weight.v[""]
        weight_sum = Vector()
        ave_weight = Vector()
        for i in range(iterations):
            print("\niter:", i, "\n")
            for j, line in enumerate(train_data):
                print("train sent: ", j)
                plain_train_sent = Sentence([tup[0] for tup in line.get_snt()])

                predict_feature = Sentence(self.tag(plain_train_sent, weight)).global_features()
                gold_feature = line.global_features()

                #predict_data = []
                #predict_data.append(Sentence(self.tag(plain_train_sent, weight)))

                if gold_feature != predict_feature:
                    weight = Vector.__iadd__(weight, Vector.__sub__(gold_feature, predict_feature))
                    #ave_weight = weight.__iadd__(gold_feature.sub(predict_feature).__rmul__(len(train_data)-i/len(train_data)))

                #print("updated weight: ", weight)
                #print(weight)
                weight_sum = Vector.__iadd__(weight_sum, weight)
                #weight_sum = weight_sum.__iadd__(weight)

            #print(j)
            ave_weight = self.average_alpha(weight_sum, i, train_data, j)
            #print("ave_weight: ", ave_weight)
            predict_dev_weight = []
            predict_dev_ave = []
            for line in dev_data:
                plain_dev_sent = Sentence([tup[0] for tup in line.get_snt()])
                predict_dev_weight.append(Sentence(self.tag(plain_dev_sent, weight)))
                predict_dev_ave.append(Sentence(self.tag(plain_dev_sent, ave_weight)))
            print("Iteration: ", i, " acc on dev with regular perceptron:", self.compute_acc(dev_data, predict_dev_weight))
            print("Iteration: ", i, " acc with avg perceptron:", self.compute_acc(dev_data, predict_dev_ave))

        return (weight, ave_weight)

    def average_alpha(self, alpha_sum, iter, train_data, j):
        avg_alpha = alpha_sum.__rmul__(1 / (len(train_data) * iter + j + 1))
        return avg_alpha

    def compute_acc(self, gold_data, predict_data):

        gold_lines = [line.get_snt() for line in gold_data]
        auto_lines = [line.get_snt() for line in predict_data]

        #print(gold_lines[-1], '\n', 123, auto_lines[-1], 456, '\n\n')
        #print(len(gold_lines[-1]), len(auto_lines[-1]))

        correct = 0.0
        total = 0.0
        for g_snt, a_snt in zip(gold_lines, auto_lines):
            #print("g_snt: ", g_snt,'\n', "a_snt: ", a_snt[1:])
            correct += sum([g_tup[1] == a_tup[1] for g_tup, a_tup in zip(g_snt, a_snt)])
            total += len(g_snt)

        return correct / total
'''
#if __name__ == "__main__":
'''
