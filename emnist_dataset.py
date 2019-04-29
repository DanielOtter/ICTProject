from emnist import list_datasets, extract_training_samples, extract_test_samples
import numpy as np

def prepare_samples(samples, label_min, label_max):
    """Prepare samples for usage.
    Filter out unwanted samples (e.g. idx_min=10, idx_max=35 for uppercase letters).
    Shuffle the label-image-pairs and cut them all to the same length for a balanced dataset.
    """
    images, labels = samples

    filtered_indices = np.where((labels >= label_min) & (labels <= label_max))
    labels = np.take(labels, filtered_indices) # filter out unwanted samples in labels
    images = np.take(images, filtered_indices, axis=0) # filter out unwanted samples in images
    labels = labels - label_min # smallest index is now 0

    labels = np.squeeze(labels)
    images = np.squeeze(images) # discard unnecessary extra dimension from take

    label_occs = np.bincount(labels) # count occurences of all labels
    smallest_label = np.argmin(label_occs) # index of least occuring label
    smallest_label_cnt = np.amin(label_occs) # count of least occuring label

    rdm_state = np.random.get_state() # save the random state of numpy
    np.random.shuffle(labels) # randomly shuffle the labels
    np.random.set_state(rdm_state) # load the previous random state
    np.random.shuffle(images) #randomly shuffle the images in the same way

    index_array = np.array([], dtype=int) #create an empty array
    for i in range(label_max-(label_min-1)): # loop through labels
        indices = np.squeeze(np.where(labels == i)) # get the indices of the current label
        indices = indices[:smallest_label_cnt] # cut the size down to the smallest
        index_array = np.append(index_array, indices) # append the indices

    labels = np.squeeze(labels[index_array])
    images = np.squeeze(images[index_array, :, :]) # balance the labels and images

    return images, labels

train_samples = prepare_samples(extract_training_samples('byclass'), 10, 36)
test_samples = prepare_samples(extract_test_samples('byclass'), 10, 36)


class structured_array():
    def __init__(self, name, sample_data):
        self.name = name
        self.images, self.labels = sample_data

# emnist_dataset is final dataset
emnist_dataset = []
# for training: emnist_dataset[0].name,emnist_dataset[0].pictures,emnist_dataset[0].labels
emnist_dataset.append(structured_array("Training", train_samples))
# for testing: emnist_dataset[1].name,emnist_dataset[1].pictures,emnist_dataset[1].labels
emnist_dataset.append(structured_array("Testing", test_samples))


# save or load data
#np.save('emnist_dataset.npy', emnist_dataset)
#emnist_dataset = np.load('emnist_dataset.npy',allow_pickle=True)
