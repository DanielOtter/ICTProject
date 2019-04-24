from emnist import extract_training_samples, extract_test_samples
from itertools import groupby

test_images, test_labels = extract_test_samples('byclass')
train_images, train_labels = extract_training_samples('byclass')

print(test_images.shape)
print(test_labels.shape)
print(test_labels[32])
print()
print(train_images.shape)
print(train_labels.shape)
print(train_labels[32])
