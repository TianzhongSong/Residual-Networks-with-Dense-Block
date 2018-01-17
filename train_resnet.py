from __future__ import print_function

import os
import numpy as np

from utils import *
from models import resnet

from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.backend as K

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir,dataset_select):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy_{}.png'.format(dataset_select)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss_{}.png'.format(dataset_select)))
    plt.close()


def save_history(history, result_dir,dataset_select):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(dataset_select)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


if __name__ == '__main__':
    
    batch_size = 64
    widen_factor = 1 # 1 for original ResNet, else for WRN
    nb_blocks = [8,8,8]  # number of blocks
    datasets = ['cifar10','cifar100','fashion-mnist','svhn']
    dataset_select = datasets[0] # 0,1,2,3 for 'cifar10','cifar100','fashion-mnist','svhn' respectively.
    result_dir = 'results/'  # dir for saving results
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if dataset_select == 'cifar10':
        nb_classes = 10
        epochs = 275
        learning_rate = 0.1
        droprate = 0.0
        print ('load cifar10 data..')
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        input_shape = (32,32,3)

        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([63.0, 62.1, 66.7])
        x_train -= mean
        x_train /= std
        x_test -= mean
        x_test /= std

    elif dataset_select == 'cifar100':
        nb_classes = 100
        epochs = 275
        learning_rate = 0.1
        droprate = 0.0
        print ('load cifar100 data..')
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        input_shape = (32, 32, 3)

        mean = np.array([129.3, 124.1, 112.4])
        std = np.array([68.2, 65.4, 70.4])
        x_train -= mean
        x_train /= std
        x_test -= mean
        x_test /= std

    elif dataset_select == 'fashion-mnist':
        nb_classes = 10
        epochs = 40
        learning_rate = 0.01
        droprate = 0.2
        print ('load fashion-mnist data..')
        (x_train, y_train), (x_test, y_test) = mnist_reader.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        x_train /= 255.
        x_test /= 255.
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

    else:
        nb_classes = 10
        epochs = 40
        learning_rate = 0.01
        droprate = 0.2
        print ('load svhn data..')
        # reference : https://github.com/robertomest/convnet-study
        train_set, valid_set, test_set = get_svhn_data.load('data/svhn')
        train_set['data'] = get_svhn_data.preprocess(train_set['data'])
        test_set['data'] = get_svhn_data.preprocess(test_set['data'])
        validation_data = (test_set['data'], test_set['labels'])
        input_shape = (32,32,3)

    print ('create residual densenet..')
    model = resnet.ResNet(nb_classes = 10,
                                    img_dim = (32,32,3),
                                    k = widen_factor,
                                    nb_blocks = nb_blocks)
    opt = SGD(lr=0.1,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()

    # training
    if dataset_select == 'cifar10' or dataset_select == 'cifar100':
        print('start training with {}'.format(dataset_select))
        print('Using real-time data augmentation..')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=5. / 32,
                                    height_shift_range=5. / 32)

        data_iter = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
        history = model.fit_generator(data_iter,
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    callbacks=[schedules.onetenth_150_200_250(learning_rate)],
                                    epochs=epochs,
                                    validation_data=(x_test, y_test))

        plot_history(history, result_dir,dataset_select)
        save_history(history, result_dir,dataset_select)
        model.save_weights(result_dir + dataset_select+'.h5')

    elif dataset_select == 'fashion-mnist':
        print ('start training with fashion-mnist..')
        print('Not uisng data augmentation..')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[schedules.onetenth_20_30(learning_rate)],
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            verbose=1)
        plot_history(history, result_dir,dataset_select)
        save_history(history, result_dir,dataset_select)
        model.save_weights(result_dir + dataset_select +'.h5')
    else:
        print('start training with svhn..')
        print('Not uisng data augmentation..')
        history=model.fit(train_set['data'], train_set['labels'],
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(validation_data),
                            callbacks=[schedules.onetenth_20_30(learning_rate)],
                            shuffle=True,
                            verbose=1)
        plot_history(history, result_dir,dataset_select)
        save_history(history, result_dir,dataset_select)
        model.save_weights(result_dir + dataset_select +'.h5')