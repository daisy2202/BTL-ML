import numpy as np
import scipy
import random

EMOTION_FOLDER = './processed_data/'


def getEmotionImage():
    print('Load emotion image..................')
    train_data = np.load(EMOTION_FOLDER + 'train.npy')
    private_test_data = np.load(EMOTION_FOLDER + 'private_test.npy')
    public_test_data = np.load(EMOTION_FOLDER + 'public_test.npy')
    print('Done !')
    return train_data, public_test_data, private_test_data


''' Data augmentation method '''


def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def center_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = int((oshape[0] - crop_shape[0]) / 2)
        nw = int((oshape[1] - crop_shape[1]) / 2)
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def random_90degrees_rotation(batch, rotations=[0, 1, 2, 3]):
    for i in range(len(batch)):
        num_rotations = random.choice(rotations)
        batch[i] = np.rot90(batch[i], num_rotations)
    return batch


def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch


def random_blur(batch, sigma_max=5.0):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


def augmentation(batch, img_size, padding=None, testing=False):
    if not testing:
        batch = random_crop(batch, (img_size, img_size), padding)
        batch = random_flip_leftright(batch)
        batch = random_rotation(batch, 10)
    else:
        batch = center_crop(batch, (img_size, img_size), padding)
        # batch = random_flip_leftright(batch)
        # batch = random_blur(batch)

    return batch
