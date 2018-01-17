from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input,concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


# reference https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet
# reference https://github.com/szagoruyko/wide-residual-networks
def conv_factory(x, nb_filter, dropout_rate=0., weight_decay=1E-4):
    x = BatchNormalization(axis=-1,epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3,3),
                      kernel_initializer='he_normal',
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def denseblock(x, growth_rate,strides=(1,1),internal_layers=4,
               dropout_rate=0., weight_decay=1E-4):
    x = Conv2D(growth_rate, (3,3),
                      kernel_initializer='he_normal',
                      padding="same",
                      strides=strides,
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    list_feat = []
    list_feat.append(x)
    for i in range(internal_layers-1):
        x = conv_factory(x,growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat,axis=-1)
    return x


def Residual_DenseNet(nb_classes, input_shape, nb_blocks=[2,2,2],
                      widen_factor=1,block_type='C', weight_decay=1E-4,dropout_rate=0.):
    """
    residual networks with dense block
    param nb_classes: num of classes
    param input_shape: input shape, support datasets include cafar10,cifar100,svhn,fashion_mnist
    param nb_blocks: number of each block
    param k: widen factor
    param internal_layers: type of residual dense block
    return: RDN model
    """
    assert len(nb_blocks) == 3,'nb_blocks must be a list with 3 params'
    if block_type == 'A':
        internal_layers = 2
    elif block_type == 'B':
        internal_layers = 3
    else:
        internal_layers = 4

    model_input = Input(shape=input_shape)
    growth_rate = 16*widen_factor
    nb_filter = 16
    # Initial convolution
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(x)

    # group 1
    x = denseblock(y, growth_rate, internal_layers=internal_layers, 
                                            dropout_rate=dropout_rate)
    y = Conv2D(internal_layers * growth_rate, (1, 1),
               kernel_initializer="he_normal",
               padding="valid",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    y = add([x, y])

    for j in range(nb_blocks[0] - 1):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
        x = Activation('relu')(x)
        x = denseblock(x, growth_rate, internal_layers=internal_layers, 
                                                dropout_rate=dropout_rate)

        y = add([x, y])

    # group 2
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    y = Activation('relu')(x)

    x = denseblock(y, 2 * growth_rate, internal_layers=internal_layers, 
                            strides=(2, 2), dropout_rate=dropout_rate)
    y = Conv2D(2 * internal_layers * growth_rate, (1, 1), strides=(2, 2),
               kernel_initializer="he_normal",
               padding="valid",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    y = add([x, y])

    for j in range(nb_blocks[1] - 1):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
        x = Activation('relu')(x)
        x = denseblock(x, 2 * growth_rate, internal_layers=internal_layers, 
                                                dropout_rate=dropout_rate)

        y = add([x, y])

    # group 3
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    y = Activation('relu')(x)

    x = denseblock(y, 4 * growth_rate, strides=(2, 2), 
                    internal_layers=internal_layers, dropout_rate=dropout_rate)
    y = Conv2D(4 * internal_layers * growth_rate, (1, 1), strides=(2, 2),
               kernel_initializer="he_normal",
               padding="valid",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    y = add([x, y])

    for j in range(nb_blocks[2] - 1):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
        x = Activation('relu')(x)
        x = denseblock(x, 4 * growth_rate, internal_layers=internal_layers, 
                                                dropout_rate=dropout_rate)

        y = add([x, y])

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    RDN = Model(input=[model_input], output=[x], name="RDN")

    return RDN
