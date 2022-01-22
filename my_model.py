import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten, SeparableConv2D

from tensorflow.keras import Model

def streesigns_model(nbr_classes):

    my_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    #x = SeparableConv2D(256, (3,3))(x)
    x = Conv2D(256, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


if __name__=='__main__':

    model = streesigns_model(10)
    