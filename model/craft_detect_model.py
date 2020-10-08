import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as kb
from tensorflow.keras.optimizers import Adam
import numpy as np

from model.custom_layers import ConvBlock, UpsampleBlock

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class Craft(Model):
    def __init__(self, config, map_num: int = 1):
        super(Craft, self).__init__()

        assert map_num in [1, 2], "Map Num 은 1 아니면 2 여야함"

        self.map_num = map_num
        self.epochs = config.epochs
        self.batch_size = config.batch_size

        self.conv_block1 = ConvBlock(64)
        self.conv_block2 = ConvBlock(128)
        self.conv_block3 = ConvBlock(256)
        self.conv_block4 = ConvBlock(512)
        self.conv_block5 = ConvBlock(512)
        self.conv_block6 = ConvBlock(512, last=True)

        self.upsample_block1 = UpsampleBlock(512)
        self.upsample_block2 = UpsampleBlock(256)
        self.upsample_block3 = UpsampleBlock(128)
        self.upsample_block4 = UpsampleBlock(64, True)

        # Classification block
        self.conv_last1 = Conv2D(32, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_last2 = Conv2D(32, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_last3 = Conv2D(16, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_last4 = Conv2D(16, 1, 1, activation='relu')

        self.conv_fcn = Conv2D(map_num, 1, 1, activation='sigmoid')

    @tf.function
    def call(self, x, training=None, mask=None):
        res1 = self.conv_block1(x)
        res2 = self.conv_block2(res1)
        res3 = self.conv_block3(res2)
        res4 = self.conv_block4(res3)
        x = self.conv_block6(res4)

        x = tf.concat([x, res4], axis=3)
        x = self.upsample_block1(x)

        x = tf.concat([x, res3], axis=3)
        x = self.upsample_block2(x)

        x = tf.concat([x, res2], axis=3)
        x = self.upsample_block3(x)

        x = tf.concat([x, res1], axis=3)
        x = self.upsample_block4(x)

        x = self.conv_last1(x)
        x = self.conv_last2(x)
        x = self.conv_last3(x)
        x = self.conv_last4(x)
        x = self.conv_fcn(x)

        if self.map_num == 1:
            x = tf.squeeze(x, axis=3)
        else:
            pass

        return x

    def compile_model(self):
        if self.map_num == 1:
            self.compile(Adam(lr=0.0001), eu_loss_region)
        elif self.map_num == 2:
            self.compile(Adam(lr=0.0001), eu_loss_both)

    # @tf.function
    def train_model(self, train_x, train_y, test_x, test_y, save_route=None):

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_route + 'best_model',
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         monitor='val_loss',
                                                         save_best_only=True)

        self.fit(train_x, train_y, validation_data=(test_x, test_y),
                 epochs=self.epochs, batch_size=self.batch_size, callbacks=[cp_callback])


def eu_loss_region(y_true, y_pred):
    return kb.sqrt(kb.sum(kb.square(y_true - y_pred), axis=-1))


def eu_loss_both(y_true, y_pred):
    return kb.sqrt(kb.sum(kb.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]), axis=-1)) + \
           kb.sqrt(kb.sum(kb.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1]), axis=-1))
