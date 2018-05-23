# Residual Networks with Dense Block

## The combination of Resnet and Densenet

![rdb](https://github.com/TianzhongSong/Residual-Networks-with-Dense-Block/blob/master/imgs/rdb.jpg)

![rdbs](https://github.com/TianzhongSong/Residual-Networks-with-Dense-Block/blob/master/imgs/rdbs.jpg)

## Training

Training supprots CIFAR-10/100, SVHN, Fashion-MNIST

A example of RNDB-50-C for CIFAR-10:

    python train_RNDB.py --data-type='cifar-10' --block-type='C' --depth=50 --lr=0.1

## Results

![c10](https://github.com/TianzhongSong/Residual-Networks-with-Dense-Block/blob/master/imgs/c10.jpg)

![c10c](https://github.com/TianzhongSong/Residual-Networks-with-Dense-Block/blob/master/imgs/c10c.jpg)

![c100](https://github.com/TianzhongSong/Residual-Networks-with-Dense-Block/blob/master/imgs/c100.jpg)

![c100c](https://github.com/TianzhongSong/Residual-Networks-with-Dense-Block/blob/master/imgs/c100c.jpg)

## Wide Residual Networks with Dense Block

A example of WRNDB-50-4-C for CIFAR-10:

    python train_RNDB.py --data-type='cifar-10' --block-type='C' --depth=50 --widen-factor=4 --lr=0.1

