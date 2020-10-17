from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,BatchNormalization, Conv2DTranspose, Reshape, Add
from keras.models import Model
from keras import backend as K
import keras

def build_celebA_model():
    w_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    g_init = keras.initializers.RandomNormal(mean=1.0, stddev=0.02)
    ef_dim = 64
    z_dim = 512
    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16) # 32,16,8,4
    gf_dim = 64

    net_input = Input(shape=(64,64,3))
    net_h0 = Conv2D(filters=ef_dim, kernel_size=(9,9),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_input)
    net_h0 = BatchNormalization(gamma_initializer=g_init)(net_h0)
    net_h1 = Conv2D(filters=ef_dim*2, kernel_size=(7, 7),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_h0)
    net_h1 = BatchNormalization(gamma_initializer=g_init)(net_h1)
    net_h2 = Conv2D(filters=ef_dim*4, kernel_size=(5, 5),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_h1)
    net_h2 = BatchNormalization(gamma_initializer=g_init)(net_h2)
    net_h3 = Conv2D(filters=ef_dim*8, kernel_size=(5, 5),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_h2)
    net_h3 = BatchNormalization(gamma_initializer=g_init)(net_h3)
    net_h4 = Flatten()(net_h3)
    net_out1 = Dense(z_dim, kernel_initializer=w_init,bias_initializer=w_init)(net_h4)
    net_out1 = BatchNormalization(gamma_initializer=g_init)(net_out1)
    net_h5 = Flatten()(net_h3)
    net_out2 = Dense(z_dim, activation='softplus',kernel_initializer=w_init)(net_h5)
    net_out2 = BatchNormalization( gamma_initializer=g_init)(net_out2)
    encoded = Add()([net_out1, net_out2])

    net_h6 = Dense(gf_dim*4*s8*s8, kernel_initializer=w_init, activation='relu')(encoded)
    net_h7 = Reshape((s8, s8, gf_dim*4))(net_h6)
    net_h7 = BatchNormalization(gamma_initializer=g_init)(net_h7)

    net_h8 = Conv2DTranspose(filters=gf_dim*4, kernel_size=(5, 5), strides=(2, 2),
                            padding='SAME', activation='relu', kernel_initializer=w_init)(net_h7)
    net_h8 = BatchNormalization(gamma_initializer=g_init)(net_h8)
    net_h9 = Conv2DTranspose(filters=gf_dim*2, kernel_size=(5, 5), strides=(2, 2),
                            padding='SAME', activation='relu', kernel_initializer=w_init)(net_h8)
    net_h9 = BatchNormalization(gamma_initializer=g_init)(net_h9)
    net_h10 = Conv2DTranspose(filters=gf_dim//2, kernel_size=(7, 7), strides=(2, 2),
                            padding='SAME', activation='relu', kernel_initializer=w_init)(net_h9)
    net_h10 = BatchNormalization(gamma_initializer=g_init)(net_h10)


    decoded = Conv2DTranspose(filters=3, kernel_size=(9, 9), strides=(1, 1),
                            padding='SAME', activation='sigmoid', kernel_initializer=w_init)(net_h10)

    return Model(net_input, decoded)


def build_mnist_model():
    w_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    g_init = keras.initializers.RandomNormal(mean=1.0, stddev=0.02)
    ef_dim = 28
    z_dim = 256
    image_size = 28 #  the output size of generator
    s2, s4, s7 = int(image_size/2), int(image_size/4), int(image_size/7)
    gf_dim = 28

    net_input = Input(shape=(28,28,1))
    net_h0 = Conv2D(filters=ef_dim, kernel_size=(5,5),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_input)
    net_h0 = BatchNormalization(gamma_initializer=g_init)(net_h0)
    net_h1 = Conv2D(filters=ef_dim*2, kernel_size=(5, 5),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_h0)
    net_h1 = BatchNormalization(gamma_initializer=g_init)(net_h1)
    net_h2 = Conv2D(filters=ef_dim*4, kernel_size=(3, 3),strides=(2, 2), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_h1)
    net_h2 = BatchNormalization(gamma_initializer=g_init)(net_h2)
    net_h3 = Conv2D(filters=ef_dim*7, kernel_size=(3, 3),strides=(1, 1), activation='relu',
                            padding='SAME', kernel_initializer=w_init)(net_h2)
    net_h3 = BatchNormalization(gamma_initializer=g_init)(net_h3)
    net_h4 = Flatten()(net_h3)
    net_out1 = Dense(z_dim, kernel_initializer=w_init,bias_initializer=w_init)(net_h4)
    net_out1 = BatchNormalization(gamma_initializer=g_init)(net_out1)
    net_h5 = Flatten()(net_h3)
    net_out2 = Dense(z_dim, activation='softplus',kernel_initializer=w_init)(net_h5)
    net_out2 = BatchNormalization( gamma_initializer=g_init)(net_out2)
    encoded = Add()([net_out1, net_out2])

    net_h6 = Dense(gf_dim*4*s7*s7, kernel_initializer=w_init, activation='relu')(encoded)
    net_h7 = Reshape((s7, s7, gf_dim*4))(net_h6)
    net_h7 = BatchNormalization(gamma_initializer=g_init)(net_h7)

    net_h8 = Conv2DTranspose(filters=gf_dim*4, kernel_size=(4, 4), strides=(1, 1),
                            padding='valid', activation='relu', kernel_initializer=w_init)(net_h7)
    net_h8 = BatchNormalization(gamma_initializer=g_init)(net_h8)
    net_h9 = Conv2DTranspose(filters=gf_dim*2, kernel_size=(3, 3), strides=(2, 2),
                            padding='SAME', activation='relu', kernel_initializer=w_init)(net_h8)
    net_h9 = BatchNormalization(gamma_initializer=g_init)(net_h9)
    net_h10 = Conv2DTranspose(filters=gf_dim//2, kernel_size=(5, 5), strides=(2, 2),
                            padding='SAME', activation='relu', kernel_initializer=w_init)(net_h9)
    net_h10 = BatchNormalization(gamma_initializer=g_init)(net_h10)


    decoded = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1, 1),
                            padding='SAME', activation='sigmoid', kernel_initializer=w_init)(net_h10)

    return Model(net_input, decoded)
