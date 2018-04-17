import chainer
import cv2
import numpy as np
import os
import progressbar
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train import get_imbalanced_mnist

def save_image(data, dirname):
    data_iter = chainer.iterators.SerialIterator(data, batch_size=1, shuffle=True, repeat=False)
    pbar = progressbar.ProgressBar(len(data))
    for i, batch in enumerate(data_iter):
        x, t = chainer.dataset.concat_examples(batch)
        im_dir = 'data/images/{}/{}'.format(dirname, t[0])
        if not os.path.exists(im_dir):
            os.system('mkdir -p {}'.format(im_dir))
        im = (x.reshape(28, 28) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(im_dir, '{}.jpg'.format(i)), im)
        pbar.update(i)

if __name__ == '__main__':
    train, val, test = get_imbalanced_mnist()
    save_image(train, 'train')
    save_image(val, 'val')
    save_image(test, 'test')
