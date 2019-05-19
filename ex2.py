import numpy as np
import csv
#from nltk.util import ngrams

# used fastest method for large arrays,
# as tested in https://stackoverflow.com/questions/44587746/length-of-each-string-in-a-numpy-array
def np_view():
    v = myarray.view(np.uint32).reshape(myarray.size, -1)
    l = np.argmin(v, 1)
    l[v[np.arange(len(v)), l] > 0] = v.shape[-1]
    return l

# taken from nltk library and changed a little for padding at n=1
from itertools import chain
def ngrams(
    sequence,
    n,
    pad=False,
    pad_symbol=None,
):
    sequence = iter(sequence)
    if pad:
        if n > 1:
            sequence = chain((pad_symbol,) * (n - 1), sequence)
            sequence = chain(sequence, (pad_symbol,) * (n - 1))
        elif n == 1:  # added for padding at n=1
            sequence = chain((pad_symbol,) * (n), sequence)
            sequence = chain(sequence, (pad_symbol,) * (n))

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def bijectivemap(N_grams, longestWord, words_by_letters ,ConvertToInt,ConvertToChars):
    N_gram_new = []
    temp = []
    for i in range(0, longestWord):  # longestWord
        if (words_by_letters[i]).size != 0:
            for j in range(0, (words_by_letters[i]).size):
                temp1 = N_grams[i][j]
                if ConvertToInt:
                    # convert to int
                    temp1 = temp1.view(np.uint32)
                    substract_ind = np.where(temp1 > 96)
                    substract_ind2 = np.where(temp1 < 96)
                    # a starts at 0
                    temp1[substract_ind] = temp1[substract_ind] - 97
                    # padding_symbol is 26
                    temp1[substract_ind2] = temp1[substract_ind2] - 69
                elif ConvertToChars:
                    temp1 = temp1.view(np.uint32)
                    substract_ind = np.where(temp1 < 26)
                    substract_ind2 = np.where(temp1 == 26)
                    # a starts at 0
                    temp1[substract_ind] = temp1[substract_ind] + 97
                    # padding_symbol is 26
                    temp1[substract_ind2] = temp1[substract_ind2] + 69
                    temp1 = np.atleast_2d(list(''.join(chr(i) for i in temp1)))
                    temp1 = np.transpose(temp1)
                temp.append(temp1)
        if (words_by_letters[i]).size != 0:
            N_gram_new.append(temp)
        else:
            N_gram_new.append([])
        temp = []
    return N_gram_new

def create_Ngrams(N_grams,words_by_letters, longestWord, degreeNgram,ConvertToInt,ConvertToChars):
    # first argument null if no array ngram wants to be converted,
    # first argument another ngram if char and array want to be swapped back again
    # convertoint and converttochars bothe need to be in EXOR relation
    temp = []
    if N_grams == []:
        N_gram = []
        for i in range(0, longestWord):  # longestWord
            if (words_by_letters[i]).size != 0:
                for j in range(0, (words_by_letters[i]).size):
                    if (words_by_letters[i]).size > 1:
                        temp1 = list(ngrams(words_by_letters[i][j], degreeNgram, pad=True, pad_symbol='_'))
                        # temp1 = list(words_by_letters[i][j])
                    else:
                        temp1 = list(ngrams(words_by_letters[i][0], degreeNgram, pad=True, pad_symbol='_'))
                    temp1 = np.array(temp1)
                    temp.append(temp1)
            if (words_by_letters[i]).size != 0:
                N_gram.append(temp)
            else:
                N_gram.append([])
            temp = []
        N_gram = bijectivemap(N_gram, longestWord, words_by_letters, ConvertToInt, ConvertToChars)
    elif N_grams != []:
        N_gram = bijectivemap(N_grams,longestWord,words_by_letters,ConvertToInt,ConvertToChars)

    return N_gram




f = open('intelligent_plastic_machines.txt', 'r')
x = f.readlines()
f.close()
data = x[0].split()
myarray = np.asarray(data)
longestWord = len(max(myarray, key=len))#weird word

word_lengths = np_view()

words_by_letters = []  # integer array init
for i in range (1,longestWord+1):
    index_letter_count = (np.where(word_lengths== i))
    #if len(index_letter_count[0])!=0:
    words_by_letters.append(np.atleast_1d(np.squeeze(np.take(myarray,index_letter_count))))

#read in csv file
data = list(csv.reader(open('confusion_matrix.csv')))

degreeNgram = 1
N_grams = []
Ngram = create_Ngrams(N_grams,words_by_letters, longestWord, degreeNgram,ConvertToInt = True,ConvertToChars=False)


b=5