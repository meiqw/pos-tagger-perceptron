'''
lines = ["In_IN an_DT Oct._NNP", "19_CD review_NN of_IN"]
lines = [[tup.split('_') for tup in line.split()] for line in lines]
print(lines)
for line in lines:
    print(line)
'''
lines_2 = ["No , it was n't Black Monday .","The equity market was illiquid ."]
lines_2 = [l.split() for l in lines_2]
print(lines_2)
for l in lines_2:
    print(l)
