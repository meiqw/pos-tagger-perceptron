from sparse_vector import Vector

class Sentence(object):
    def __init__(self, snt):
        ''' Modify if necessary.
        '''
        self.snt = snt
        #print(self.snt)
        return

    def global_features(self):
        ''' Implement your feature extraction code here. This takes annotated or unannotated sentence
        and return a set of features
        '''
        sent_features = Vector()
        for i, element in enumerate(self.snt):
            if len(element) == 2:
                word = element[0]
                tag = element[1]

                #cur_word = 'word={}'.format(word)
                #sent_features.v[cur_word] += 1

                #emission
                emission = 'word={} tag={}'.format(word,tag)
                #print(emission)
                sent_features.v[emission] += 1

                pre = word[:3] if len(word) >= 3 else ""
                prefix = 'prefix={} tag={}'.format(pre, tag)
                sent_features.v[prefix] += 1
                su = word[-4] if len(word) >= 4 else ""
                suffix = 'suffix={} tag={}'.format(su, tag)
                sent_features.v[suffix] += 1

                word_1 = self.snt[i - 2][0] if i - 2 > 0 else ""
                word_1_feature = 'word_1={} tag={}'.format(word_1, tag)
                word_2 = self.snt[i - 1][0] if i - 1 > 0 else ""
                word_2_feature = 'word_2={} tag={}'.format(word_2, tag)

                word1 = self.snt[i + 1][0] if i + 1 < len(self.snt) else ""
                word1_feature = 'word1={} tag={}'.format(word1, tag)
                word2 = self.snt[i + 2][0] if i + 2 < len(self.snt) else ""
                word2_feature = 'word2={} tag={}'.format(word2, tag)

                if i == 0:
                    word_2 = "START"
                    word_2_feature = 'word_2={} tag={}'.format(word_2, tag)
                if i == 1:
                    word_1 = "START"
                    word_1_feature = 'word_1={} tag={}'.format(word_1, tag)

                if i == len(self.snt)-2:
                    word2 = "$END"
                    word2_feature = 'word2={} tag={}'.format(word2, tag)
                if i == len(self.snt)-1:
                    word1 = "$END"
                    word1_feature = 'word1={} tag={}'.format(word1, tag)

                sent_features.v[word_1_feature] += 1
                sent_features.v[word_2_feature] += 1
                sent_features.v[word1_feature] += 1
                sent_features.v[word2_feature] += 1

                #transition
                if i == 0:
                    transition = 'prev_tag={} curr_tag={}'.format('<S>', tag)
                else:
                    prev_tag = self.snt[i-1][1]
                    transition = 'prev_tag={} curr_tag={}'.format(prev_tag, tag)
                sent_features.v[transition] += 1

        transition = 'prev_tag={} curr_tag={}'.format(self.snt[-1][1], "</S>")
        sent_features.v[transition] += 1

        return sent_features

    def local_feature(self, cur_word, prev_tag, cur_tag, word_1, word_2, word1, word2):
        sent_features = Vector()
        emission = 'word={} tag={}'.format(cur_word, cur_tag)
        sent_features.v[emission] = 1
        transition = 'prev_tag={} curr_tag={}'.format(prev_tag, cur_tag)
        sent_features.v[transition] = 1

        word_1_feature = 'word_1={} tag={}'.format(word_1, cur_tag)
        word_2_feature = 'word_2={} tag={}'.format(word_2, cur_tag)
        sent_features.v[word_1_feature] = 1
        sent_features.v[word_2_feature] = 1

        word1_feature = 'word1={} tag={}'.format(word1, cur_tag)
        word2_feature = 'word2={} tag={}'.format(word2, cur_tag)
        sent_features.v[word1_feature] = 1
        sent_features.v[word2_feature] = 1

        pre = cur_word[:3] if len(cur_word) >= 3 else ""
        prefix = 'prefix={} tag={}'.format(pre, cur_tag)
        sent_features.v[prefix] = 1
        su = cur_word[-4] if len(cur_word) >= 4 else ""
        suffix = 'suffix={} tag={}'.format(su, cur_tag)
        sent_features.v[suffix] += 1

        return sent_features

    def get_snt(self):
        return self.snt
