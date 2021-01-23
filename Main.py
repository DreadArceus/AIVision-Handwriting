from ImgProcessing import process_images, process_labels
from Model import train

file_paths = {
    'images': './Dataset/train-images-idx3-ubyte',
    'labels': './Dataset/train-labels-idx1-ubyte',
    'test-images': './Dataset/t10k-images-idx3-ubyte',
    'test-labels': './Dataset/t10k-labels-idx1-ubyte',
}

train_images = process_images(file_paths['images'])
train_labels = process_labels(file_paths['labels'])

train(train_images, train_labels)

test_images = process_images(file_paths['test-images'])
test_labels = process_labels(file_paths['test-labels'])
