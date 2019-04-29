from emnist import list_datasets, extract_training_samples, extract_test_samples
import numpy as np
import random
import array
#import matplotlib.pyplot as plt    #for debugging

def prepare_samples(samples, label_min, label_max):
    # get images and labels
    images, labels = samples

    index_uppercase = np.where((labels >= label_min) & (labels <= label_max))  # filter out needed uppercase letters
    labels = np.squeeze(np.take(labels, index_uppercase))
    images = np.squeeze(images[index_uppercase[:], :, :])
    labels = labels - label_min  # now indices are from 0 to 25

    label_values = np.bincount(labels)  # count occurences of all labels
    #smallest_label = np.argmin(label_values)  # index of least occurent label
    smallest_label_count = np.amin(label_values)  # count of least occurent label

    # pick smallest_label_value amount
    index = array.array('i')  # integer array init
    for x in range(label_max-(label_min-1)):
        ind = np.squeeze(np.where(labels == x))
        random.shuffle(ind)#to not take the first letters
        ind = ind[:smallest_label_count]
        index = np.append(index, ind)

    random.shuffle(index)
    labels = np.take(labels, index)
    images = np.squeeze(images[index, :, :])

    return images, labels


train_samples = prepare_samples(extract_training_samples('byclass'), 10, 35)
test_samples = prepare_samples(extract_test_samples('byclass'), 10, 35)

class structured_array():
    def __init__(self, name, sample_data):
        self.name = name
        self.images, self.labels = sample_data

# emnist_dataset is final dataset
emnist_dataset = []
# for training: emnist_dataset[0].name,emnist_dataset[0].images,emnist_dataset[0].labels
emnist_dataset.append(structured_array("Training", train_samples))
# for testing: emnist_dataset[1].name,emnist_dataset[1].images,emnist_dataset[1].labels
emnist_dataset.append(structured_array("Testing", test_samples))

# debbuging
#plt.imshow(emnist_dataset[0].images[6, :, :], cmap="gray")
#plt.show()

# save or load data
#np.save('emnist_dataset.npy', emnist_dataset)
#emnist_dataset = np.load('emnist_dataset.npy',allow_pickle=True)