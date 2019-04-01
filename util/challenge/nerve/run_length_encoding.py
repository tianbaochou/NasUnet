'''
Fast inplementation of Run-Length Encoding algorithm
Takes only 200 seconds to process 5635 mask files
'''
import numpy as np
from PIL import Image
import os
from util.utils import create_exp_dir

def rle_encoding(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 5:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

if __name__ == '__main__':
    input_path = '../../../predictions/nerve_rst'
    masks = [f for f in os.listdir(input_path) if f.endswith('_mask.tif')]
    masks = sorted(masks, key=lambda d:int(d.split('_')[0]))

    encodings = []
    total = len(masks)
    for m in masks:
        img = Image.open(os.path.join(input_path, m))
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        print('----->: processing: {}'.format(m))
        encodings.append(rle_encoding(x))
    print('Encode done, write to submission file')
    # check output
    first_row = 'img,pixels'
    create_exp_dir('./submission', '=> create submission file')
    file_name = os.path.join('submission', 'submission.csv')
    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(i+1) + ',' + encodings[i]
            f.write(s + '\n')
    f.close()
    print('write done!')
