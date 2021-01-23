import numpy as np
import struct

file_paths = {
    'images': './Dataset/train-images-idx3-ubyte',
    'labels': 'train-labels-idx1-ubyte',
}

train_img_idx = open(file_paths['images'], 'rb')
train_img_idx.seek(0)

magic_num = struct.unpack('>2BH', train_img_idx.read(4))[2]
img_count = struct.unpack('>I', train_img_idx.read(4))[0]
row_count = struct.unpack('>I', train_img_idx.read(4))[0]
column_count = struct.unpack('>I', train_img_idx.read(4))[0]

for i in range(10):
    print(f'Image number {i}:')
    for x in range(row_count):
        for y in range(column_count):
            pixel = struct.unpack('>B', train_img_idx.read(1))[0]
            if pixel < 100:
                print('0', end='')
            if pixel < 10:
                print('0', end='')
            print(pixel, end=' ')
        print()

print(magic_num, img_count, row_count, column_count)