from emnist import list_datasets, extract_training_samples, extract_test_samples
import numpy as np
import random
import array
#import matplotlib.pyplot as plt    #for debugging


# get train_images and labels

images_train, labels_train = extract_training_samples('byclass')

index_uppercase = np.where((labels_train >= 10) & (labels_train <= 35))  # filter out needed uppercase letters
labels_train = np.squeeze(np.take(labels_train, index_uppercase))
images_train = np.squeeze(images_train[index_uppercase[:], :, :])
labels_train = labels_train - 10  # now indices are from 0 to 25

label_values = np.bincount(labels_train)  # count occurences of all labels
#smallest_label = np.argmin(label_values)  # index of least occurent label
smallest_label_count = np.amin(label_values)  # count of least occurent label

# pick smallest_label_value amount
train_index = array.array('i')  # integer array init
for x in range(0, 26):
    ind = np.squeeze(np.where(labels_train == x))
    random.shuffle(ind)#to not take the first letters
    ind = ind[:smallest_label_count]
    train_index = np.append(train_index, ind)

random.shuffle(train_index)
train_label = np.take(labels_train, train_index)
bilder_train = np.squeeze(images_train[train_index, :, :])



# get test_images and labels (test data is roughly 1/6 of train data,
# checked by comparing smallest_label_value and smallest_label_value_test)

images_test, labels_test = extract_test_samples('byclass')

index_uppercase = np.where((labels_test >= 10) & (labels_test <= 35))  # filter out needed uppercase letters
labels_test = np.squeeze(np.take(labels_test, index_uppercase))
images_test = np.squeeze(images_test[index_uppercase[:], :, :])
labels_test = labels_test - 10  # now indices are from 0 to 25

label_values = np.bincount(labels_test)  # count occurences of all labels
#smallest_label = np.argmin(label_values)  # index of least occurent label
smallest_label_count_test = np.amin(label_values)  # count of least occurent label

# pick smallest_label_value amount
test_index = array.array('i')  # integer array init
for x in range(0, 26):
    ind = np.squeeze(np.where(labels_test == x))
    random.shuffle(ind)#to not take the first letters
    ind = ind[:smallest_label_count_test]
    test_index = np.append(test_index, ind)

random.shuffle(test_index)
test_label = np.take(labels_test, test_index)
bilder_test = np.squeeze(images_test[test_index, :, :])

#debbuging
#plt.imshow(bilder_test[0, :, :], cmap="gray")
#plt.show()

class structured_array():
    def __init__(self, name, picture_data,labels):
        self.name = name
        self.pictures = picture_data
        self.labels = labels

# emnist_dataset is final dataset
emnist_dataset = []
# for training: emnist_dataset[0].name,emnist_dataset[0].pictures,emnist_dataset[0].labels
emnist_dataset.append(structured_array("Training", bilder_train,train_label))
# for testing: emnist_dataset[1].name,emnist_dataset[1].pictures,emnist_dataset[1].labels
emnist_dataset.append(structured_array("Testing", bilder_test,test_label))


# save or load data
np.save('emnist_dataset.npy', emnist_dataset)
#emnist_dataset = np.load('emnist_dataset.npy',allow_pickle=True)

