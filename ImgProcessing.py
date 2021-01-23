import numpy as np
import struct
# from matplotlib import pyplot


def process_images(file_path: str) -> np.ndarray:
    img_idx = open(file_path, 'rb')
    img_idx.seek(0)
    magic_num = struct.unpack('>4B', img_idx.read(4))
    img_count: int = struct.unpack('>I', img_idx.read(4))[0]
    row_count: int = struct.unpack('>I', img_idx.read(4))[0]
    column_count: int = struct.unpack('>I', img_idx.read(4))[0]
    images: np.ndarray = np.fromfile(
        img_idx, dtype=np.dtype(np.ubyte).newbyteorder('>'))
    images = images.reshape(img_count, row_count, column_count)
    return images


def process_labels(file_path: str) -> np.ndarray:
    lb_idx = open(file_path, 'rb')
    lb_idx.seek(0)
    magic_num = struct.unpack('>4B', lb_idx.read(4))
    lb_count: int = struct.unpack('>I', lb_idx.read(4))[0]
    labels: np.ndarray = np.fromfile(
        lb_idx, dtype=np.dtype(np.ubyte).newbyteorder('>'))
    labels = labels.reshape(lb_count)
    return labels


# print(f'Image number {img_count-1}: {images[img_count-1]}')
# print(magic_num, img_count, row_count, column_count)
# print(train_labels[7])
# pyplot.imshow(train_images[7], cmap='gray')
# pyplot.show()
