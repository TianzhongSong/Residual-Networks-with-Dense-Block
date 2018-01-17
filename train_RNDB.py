# -*- coding:utf-8 -*-
import argparse
import os
import numpy as np

from utils import *
from models import rndb

from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.backend as K

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir,name):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy_{}.png'.format(name)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss_{}.png'.format(name)))
    plt.close()


def save_history(history, result_dir,name):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(name)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def main():
    if args.depth == 26:
        if args.block_type == 'A':
            units = [4, 4, 4]
        elif args.block_type == 'B':
            units = [2, 3, 3]
        elif args.block_type == 'C':
            units = [2, 2, 2]
        else:
            raise ValueError("block type must be A, B and C")        
    elif args.depth == 50:
        if args.block_type == 'A':
            units = [8, 8, 8]
        elif args.block_type == 'B':
            units = [5, 6, 5]
        elif args.block_type == 'C':
            units = [4, 4, 4]
        else:
            raise ValueError("block type must be A, B and C")
    elif args.depth == 110:
        if args.block_type == 'A':
            units = [18, 18, 18]
        elif args.block_type == 'B':
            units = [12, 12, 12]
        elif args.block_type == 'C':
            units = [9, 9, 9]
        else:
            raise ValueError("block type must be A, B and C")
    elif args.depth == 290:
        if args.block_type == 'B':
            units = [32,32,32]
        else:
            raise ValueError("no experiments done on depth {} with block type {}".format(args.depth,args.block_type))
    elif args.depth == 400:
        if args.block_type == 'A':
            units = [66,67,66]
        else:
            raise ValueError("no experiments done on depth {} with block type {}".format(args.depth,args.block_type))
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))

    nb_classes = 100 if args.data_type == 'cifar-100' else 10
    input_shape = (28,28,1) if args.data_type == 'fashion-mnist' else (32,32,3)
    print("create RNDB model..")
    model = rndb.Residual_DenseNet(nb_classes = nb_classes,
                                    input_shape = input_shape,
                                    widen_factor = args.widen_factor,
                                    nb_blocks = units,
                                    block_type=args.block_type,
                                    dropout_rate = args.drop_rate)
    opt = SGD(lr=args.lr,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    if args.widen_factor == 1:
        save_name = 'RNDB_'
    else:
        save_name = 'WRNDB_'

    if args.data_type == 'cifar-10':
        print ('load cifar10 data..')
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([63.0, 62.1, 66.7])
        x_train -= mean
        x_train /= std
        x_test -= mean
        x_test /= std

        print('start training on cifar-10')
        print('Using real-time data augmentation..')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=5. / 32,
                                    height_shift_range=5. / 32)
        data_iter = datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=True)
        history = model.fit_generator(data_iter,
                                    steps_per_epoch=x_train.shape[0] // args.batch_size,
                                    callbacks=[schedules.onetenth_150_200_250(args.lr)],
                                    epochs=args.num_epoch,
                                    validation_data=(x_test, y_test))

        plot_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        save_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        model.save_weights(args.save_dir + save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type+'.h5')

    elif args.data_type == 'cifar-100':
        print('load cifar100 data..')
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        mean = np.array([129.3, 124.1, 112.4])
        std = np.array([68.2, 65.4, 70.4])
        x_train -= mean
        x_train /= std
        x_test -= mean
        x_test /= std
        print('start training on cifar-100..')
        print('Using real-time data augmentation..')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=5. / 32,
                                    height_shift_range=5. / 32)
        data_iter = datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=True)
        history = model.fit_generator(data_iter,
                                    steps_per_epoch=x_train.shape[0] // args.batch_size,
                                    callbacks=[schedules.onetenth_150_200_250(args.lr)],
                                    epochs = args.num_epoch,
                                    validation_data = (x_test, y_test))

        plot_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        save_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        model.save_weights(args.save_dir + save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type+'.h5')
    elif args.data_type == 'fashion-mnist':
        print ('load fashion-mnist data..')
        (x_train, y_train), (x_test, y_test) = mnist_reader.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        x_train /= 255.
        x_test /= 255.
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        print('start training on fashion-mnist..')
        print('Not uisng data augmentation..')
        history = model.fit(x_train, y_train,
                            batch_size=args.batch_size,
                            epochs=args.num_epoch,
                            callbacks=[schedules.onetenth_20_30(args.lr)],
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            verbose=1)
        plot_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        save_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        model.save_weights(args.save_dir + save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type+'.h5')
    elif args.data_type == 'svhn':
        print ('load svhn data..')
        # reference : https://github.com/robertomest/convnet-study
        train_set, valid_set, test_set = get_svhn_data.load('data/svhn')
        train_set['data'] = get_svhn_data.preprocess(train_set['data'])
        test_set['data'] = get_svhn_data.preprocess(test_set['data'])
        validation_data = (test_set['data'], test_set['labels'])
        print('start training on svhn..')
        print('Not uisng data augmentation..')
        history=model.fit(train_set['data'], train_set['labels'],
                            batch_size=args.batch_size,
                            epochs=args.num_epoch,
                            validation_data=(validation_data),
                            callbacks=[schedules.onetenth_20_30(args.lr)],
                            shuffle=True,
                            verbose=1)
        plot_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        save_history(history, args.save_dir,save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type)
        model.save_weights(args.save_dir + save_name+str(args.depth)+'_'+str(args.block_type)+'_'+args.data_type+'.h5')
    else:
        raise ValueError("only support cifar-10, cifar-100, fashion-mnist and svhn dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='command for training RNDB')
    parser.add_argument('--data-type',     type=str, default='cifar-10',help='support cifar-10, cifar-100, fashion-mnist and svhn')
    parser.add_argument('--block-type',    type=str, default='A',help='A,B and C for residual dense block-A,B and C respectively')
    parser.add_argument('--depth',         type=int, default=26,help='the depth of RNDB')
    parser.add_argument('--widen-factor',  type=int, default=1,help='the widen factor for WRNDB')
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--drop-rate',     type=float, default=0.0)
    parser.add_argument('--lr',            type=float, default=0.1,help='the initial learning rate')
    parser.add_argument('--save-dir', type=str, default='./results', help='total number of train epochs')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    main()