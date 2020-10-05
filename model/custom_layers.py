import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Dropout


class ConvBlock(tf.keras.layers.Layer):
    """Layer Block with Convolutional layer and Maxpooling

    Two Convolutional Layers following a maxpooling layer
    If it is a last convolutional block, skip maxpooling layer

    Args:
          units (int) : parameter for Conv2D layer units
          last (bool) : Indicate whether its last convolutional block or not

    Returns:
        Output from layers

    """

    def __init__(self, units=64, last: bool = False):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv2D(units, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv2 = Conv2D(units, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.maxpool = MaxPooling2D()
        self.last = last

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.last is False:
            x = self.maxpool(x)
            return x
        else:
            return x


class UpsampleBlock(tf.keras.layers.Layer):
    """Upsampling Block with Convolutional layer, Batch Normalization layer and Upsampling layer

    Two Convolutional Layer with Batch Normalization Layer following a upsampling layer.
    Up sizing image which is convoluted by ConvBlock

    """
    def __init__(self, units=64, last=False):
        super(UpsampleBlock, self).__init__()

        self.upconv1 = Conv2D(units, 1, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.batch1 = BatchNormalization()
        self.upconv2 = Conv2D(units / 2, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.batch2 = BatchNormalization()
        self.upsample = UpSampling2D((2, 2))
        self.last = last

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        x = self.upconv1(inputs)
        x = self.batch1(x)
        x = self.upconv2(x)
        x = self.batch2(x)
        if self.last is True:
            return x
        else:
            return self.upsample(x)
