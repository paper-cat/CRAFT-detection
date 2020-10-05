import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from model.custom_layers import ConvBlock, UpsampleBlock


class Craft(Model):
    def __init__(self, map_num: int = 1):
        super(Craft, self).__init__()

        assert map_num in [1, 2], "Map Num 은 1 아니면 2 여야함"

        self.map_num = map_num

        self.conv_block1 = ConvBlock(64)
        self.conv_block2 = ConvBlock(128)
        self.conv_block3 = ConvBlock(256)
        self.conv_block4 = ConvBlock(512)
        self.conv_block5 = ConvBlock(512)
        self.conv_block6 = ConvBlock(512, last=True)

        self.upsample_block1 = UpsampleBlock(1024)
        self.upsample_block2 = UpsampleBlock(512)
        self.upsample_block3 = UpsampleBlock(256)
        self.upsample_block4 = UpsampleBlock(128)
        self.upsample_block5 = UpsampleBlock(64, True)

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
        x = self.upsample_block5(x)

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
