from emnist import list_datasets, extract_training_samples, extract_test_samples
import numpy as np


def prepare_samples(samples, label_min, label_max):
    images, labels = samples

    # needed_samples = np.where((labels >= label_min) & (labels <= label_max))
    # print(needed_samples)

    # labels = np.take(labels, needed_samples)
    # images = np.take(images, needed_samples, axis=0)

    # labels = np.squeeze(labels)
    # images = np.squeeze(images)

    predicate = (labels >= label_min) & (labels <= label_max)
    labels = np.compress(predicate, labels)
    images = np.compress(predicate, images, axis=0)
    labels = labels - label_min
    new_max = label_max - label_min
    new_min = 0

    label_count = np.bincount(labels)
    smallest_label_count = np.amin(label_count)

    random_state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(random_state)
    np.random.shuffle(images)

    for i in range(label_max - label_min):
        index_counter = 0
        for j in range(labels.size - 1):
            if labels[j] == i:
                index_counter += 1
            if index_counter > smallest_label_count:
                labels[j] = new_max + 1

    new_predicate = ((labels >= new_min) & (labels <= new_max))
    labels = np.compress(new_predicate, labels)
    images = np.compress(new_predicate, images, axis=0)

    # print(labels)
    # print(images)
    return images, labels


train_samples = extract_training_samples('byclass')
test_samples = extract_test_samples('byclass')

# print(train_samples)
# print(test_samples)

train_samples = prepare_samples(train_samples, 10, 35)
test_samples = prepare_samples(test_samples, 10, 35)


class structured_Array():
    def __init__(self, name, sample_data):
        self.name = name
        self.images, self.labels = sample_data


dataset = []
dataset.append(structured_Array("Training", train_samples))
dataset.append(structured_Array("Testing", test_samples))

print(dataset)

# np.save('emnist_dataset.npy', dataset)
