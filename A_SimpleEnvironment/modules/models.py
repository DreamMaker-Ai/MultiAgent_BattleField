from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Lambda, Reshape, LSTM, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Add, Concatenate, Multiply, Activation, MaxPooling2D, AveragePooling2D
from tensorflow.keras.utils import plot_model

import tensorflow.keras.backend as K


class MyConv2DModel_v0(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        layer_2 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_1)
        layer_3 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_2)
        layer_4 = Dense(units=64, activation='relu')(layer_3)
        layer_5 = Dense(units=32, activation='relu')(layer_4)
        layer_out = Dense(units=num_outputs, activation=None)(layer_5)

        """ Value network """
        val_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        val_2 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_1)
        val_3 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_2)
        val_4 = Dense(units=64, activation='relu')(val_3)
        val_5 = Dense(units=32, activation='relu')(val_4)
        val_6 = Dense(units=1, activation=None)(val_5)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_6)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv1DModel_v3A(TFModelV2):
    """
    Not yet implemented !!!!!
    MyConv1DModel_v1 の Conv1D 部分を、Policy net と Value net で共有
    共有したほう方が学習が速いかと思ったのだが収束しない！
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv1DModel_v3A, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        """
        obs_space.shape:(10,3),   action_space.n:3,   num_outputs:3
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        layer_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_1)
        layer_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (layer_2)
        layer_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_3)
        layer_5 = Dense(units=64, activation='relu')(layer_4)
        layer_6 = Dense(units=32, activation='relu')(layer_5)
        layer_out = Dense(units=num_outputs, activation=None)(layer_6)

        """ Value network """
        val_5 = Dense(units=64, activation='relu')(layer_4)
        val_6 = Dense(units=32, activation='relu')(val_5)
        val_7 = Dense(units=1, activation=None)(val_6)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=1), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv2DModel_v0B(TFModelV2):
    """
    Add Dropout layer & Batch Normalization layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0B, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        layer_bn_1 = BatchNormalization()(layer_1)
        layer_2 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_bn_1)
        layer_bn_2 = BatchNormalization()(layer_2)
        layer_3 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_bn_2)

        layer_4 = Dense(units=64, activation='relu')(layer_3)
        layer_5 = Dropout(rate=0.3)(layer_4)
        layer_6 = Dense(units=32, activation='relu')(layer_5)
        layer_out = Dense(units=num_outputs, activation=None)(layer_6)

        """ Value network """
        val_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        val_bn_1 = BatchNormalization()(val_1)
        val_2 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_bn_1)
        val_bn_2 = BatchNormalization()(val_2)
        val_3 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_bn_2)

        val_4 = Dense(units=64, activation='relu')(val_3)
        val_5 = Dropout(rate=0.3)(val_4)
        val_6 = Dense(units=32, activation='relu')(val_5)
        val_7 = Dense(units=1, activation=None)(val_6)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv2DModel_v0B_Small(TFModelV2):
    """
    Smaller Netowrk
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0B_Small, self).__init__(obs_space, action_space, num_outputs, model_config,
                                                      name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        layer_bn_1 = BatchNormalization()(layer_1)
        layer_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_bn_1)
        layer_bn_2 = BatchNormalization()(layer_2)
        layer_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_bn_2)

        layer_4 = Dense(units=32, activation='relu')(layer_3)
        layer_5 = Dropout(rate=0.3)(layer_4)
        layer_6 = Dense(units=16, activation='relu')(layer_5)
        layer_out = Dense(units=num_outputs, activation=None)(layer_6)

        """ Value network """
        val_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        val_bn_1 = BatchNormalization()(val_1)
        val_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_bn_1)
        val_bn_2 = BatchNormalization()(val_2)
        val_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_bn_2)

        val_4 = Dense(units=32, activation='relu')(val_3)
        val_5 = Dropout(rate=0.3)(val_4)
        val_6 = Dense(units=16, activation='relu')(val_5)
        val_7 = Dense(units=1, activation=None)(val_6)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv1DModel_v3C(TFModelV2):
    """
    Not yet implemented !!!!!
    Bigger & Deeper network
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv1DModel_v3C, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        """
        obs_space.shape:(10,3),   action_space.n:3,   num_outputs:3
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (self.inputs)
        layer_bn = BatchNormalization()(layer_1)
        layer_2 = Conv1D(filters=256, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn)
        layer_3 = Conv1D(filters=256, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_2)
        layer_4 = Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (layer_3)
        layer_5 = Dense(units=256, activation='relu')(layer_4)
        layer_6 = Dense(units=128, activation='relu')(layer_5)
        layer_7 = Dropout(rate=0.3)(layer_6)
        layer_8 = Dense(units=64, activation='relu')(layer_7)
        layer_out = Dense(units=num_outputs, activation=None)(layer_8)

        """ Value network """
        val_1 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (self.inputs)
        val_bn = BatchNormalization()(val_1)
        val_2 = Conv1D(filters=256, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (val_bn)
        val_3 = Conv1D(filters=256, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (val_2)
        val_4 = Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (val_3)
        val_5 = Dense(units=256, activation='relu')(val_4)
        val_6 = Dense(units=128, activation='relu')(val_5)
        val_7 = Dropout(rate=0.3)(val_6)
        val_8 = Dense(units=64, activation='relu')(val_7)
        val_9 = Dense(units=1, activation=None)(val_8)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_9)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=1), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv1DModel_v3_Simple(TFModelV2):
    """
    Not yet implemented !!!!!
    Simpler architecture
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv1DModel_v3_Simple, self).__init__(obs_space, action_space, num_outputs, model_config,
                                                      name)
        """
        obs_space.shape:(10,3),   action_space.n:3,   num_outputs:3
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (self.inputs)
        layer_2 = Conv1D(filters=256, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (layer_1)
        layer_3 = Conv1D(filters=128, kernel_size=4, strides=3, activation='relu', padding='valid') \
            (layer_2)
        layer_4 = Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', padding='valid') \
            (layer_3)
        layer_5 = Dense(units=32, activation='relu')(layer_4)
        layer_out = Dense(units=num_outputs, activation=None)(layer_5)

        """ Value network """
        val_1 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (self.inputs)
        val_2 = Conv1D(filters=256, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (val_1)
        val_3 = Conv1D(filters=128, kernel_size=4, strides=3, activation='relu', padding='valid') \
            (val_2)
        val_4 = Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', padding='valid') \
            (val_3)
        val_5 = Dense(units=32, activation='relu')(val_4)
        val_6 = Dense(units=1, activation=None)(val_5)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_6)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=1), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def ChannelAttentionModule(input: tf.keras.Model, ratio=8):
    """
    CBAM: Convolutional Block Attention Module
    Define Channel Attention Modules for 1D image
    ref. https://cocoinit23.com/keras-channel-attention-spatial-attention/
    """
    channel = input.shape[-1]
    shared_dense_1 = Dense(units=channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')
    shared_dense_2 = Dense(units=channel,
                           activation=None,
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')

    avg_pooling = GlobalAveragePooling2D()(input)
    avg_pooling = Reshape((1, 1, channel))(avg_pooling)
    avg_pooling = shared_dense_1(avg_pooling)
    avg_pooling = shared_dense_2(avg_pooling)

    max_pooling = GlobalMaxPooling2D()(input)
    max_pooling = Reshape((1, 1, channel))(max_pooling)
    max_pooling = shared_dense_1(max_pooling)
    max_pooling = shared_dense_2(max_pooling)

    x = Add()([avg_pooling, max_pooling])
    x = Activation('sigmoid')(x)

    out = Multiply()([input, x])
    return out


def SpatialAttentionModule(input: tf.keras.Model, kernel_size=7):
    """
    CBAM: Convolutional Block Attention Module
    Define Spatial Attention Modules for 1D image
    ref. https://cocoinit23.com/keras-channel-attention-spatial-attention/
    """
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input)
    x = Concatenate(axis=3)([avg_pool, max_pool])

    x = Conv2D(filters=1,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               activation='sigmoid',
               kernel_initializer='he_normal',
               use_bias=False)(x)

    out = Multiply()([input, x])
    return out


class MyConv2DModel_v0B_Small_CBAM(TFModelV2):
    """
    Add CBAM (Convolutional Block Attention Module) to MyConv2DModel_v0B_Small
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0B_Small_CBAM, self).__init__(obs_space, action_space, num_outputs, model_config,
                                                           name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        channel_attention_1 = ChannelAttentionModule(layer_1)
        spatial_attention_1 = SpatialAttentionModule(channel_attention_1, kernel_size=3)
        res_1 = Add()([layer_1, spatial_attention_1])
        layer_1_bn = BatchNormalization()(res_1)

        layer_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_1_bn)
        channel_attention_2 = ChannelAttentionModule(layer_2)
        spatial_attention_2 = SpatialAttentionModule(channel_attention_2, kernel_size=2)
        res_2 = Add()([layer_2, spatial_attention_2])
        layer_2_bn = BatchNormalization()(res_2)

        layer_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_2_bn)

        layer_4 = Dense(units=32, activation='relu')(layer_3)
        layer_5 = Dropout(rate=0.3)(layer_4)
        layer_6 = Dense(units=16, activation='relu')(layer_5)
        layer_out = Dense(units=num_outputs, activation=None)(layer_6)

        """ Value network """
        val_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        val_channel_attention_1 = ChannelAttentionModule(val_1)
        val_spatial_attention_1 = SpatialAttentionModule(val_channel_attention_1, kernel_size=3)
        val_res_1 = Add()([val_1, val_spatial_attention_1])
        val_1_bn = BatchNormalization()(val_res_1)

        val_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_1_bn)
        val_channel_attention_2 = ChannelAttentionModule(val_2)
        val_spatial_attention_2 = SpatialAttentionModule(val_channel_attention_2, kernel_size=2)
        val_res_2 = Add()([val_2, val_spatial_attention_2])
        val_2_bn = BatchNormalization()(val_res_2)

        val_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_2_bn)

        val_4 = Dense(units=32, activation='relu')(val_3)
        val_5 = Dropout(rate=0.3)(val_4)
        val_6 = Dense(units=16, activation='relu')(val_5)
        val_7 = Dense(units=1, activation=None)(val_6)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv2DModel_v0B_Small_CBAM_1DConv(TFModelV2):
    """
    Add CBAM (Convolutional Block Attention Module) to MyConv2DModel_v0B_Small
    Add 1x1 Conv to policy network and value network without weight share
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0B_Small_CBAM_1DConv, self).__init__(obs_space, action_space, num_outputs,
                                                                  model_config,
                                                                  name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_0 = Conv2D(filters=16, kernel_size=1, activation='relu')(self.inputs)
        # layer_0 = Conv2D(filters=16, kernel_size=1, activation='tanh')(self.inputs)
        layer_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (layer_0)
        channel_attention_1 = ChannelAttentionModule(layer_1)
        spatial_attention_1 = SpatialAttentionModule(channel_attention_1, kernel_size=3)
        res_1 = Add()([layer_1, spatial_attention_1])
        layer_1_bn = BatchNormalization()(res_1)

        layer_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_1_bn)
        channel_attention_2 = ChannelAttentionModule(layer_2)
        spatial_attention_2 = SpatialAttentionModule(channel_attention_2, kernel_size=2)
        res_2 = Add()([layer_2, spatial_attention_2])
        layer_2_bn = BatchNormalization()(res_2)

        layer_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_2_bn)

        layer_4 = Dense(units=32, activation='relu')(layer_3)
        layer_5 = Dropout(rate=0.3)(layer_4)
        layer_6 = Dense(units=16, activation='relu')(layer_5)
        layer_out = Dense(units=num_outputs, activation=None)(layer_6)

        """ Value network """
        val_0 = Conv2D(filters=16, kernel_size=1, activation='relu')(self.inputs)
        # val_0 = Conv2D(filters=16, kernel_size=1, activation='tanh')(self.inputs)
        val_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (val_0)
        val_channel_attention_1 = ChannelAttentionModule(val_1)
        val_spatial_attention_1 = SpatialAttentionModule(val_channel_attention_1, kernel_size=3)
        val_res_1 = Add()([val_1, val_spatial_attention_1])
        val_1_bn = BatchNormalization()(val_res_1)

        val_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_1_bn)
        val_channel_attention_2 = ChannelAttentionModule(val_2)
        val_spatial_attention_2 = SpatialAttentionModule(val_channel_attention_2, kernel_size=2)
        val_res_2 = Add()([val_2, val_spatial_attention_2])
        val_2_bn = BatchNormalization()(val_res_2)

        val_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_2_bn)

        val_4 = Dense(units=32, activation='relu')(val_3)
        val_5 = Dropout(rate=0.3)(val_4)
        val_6 = Dense(units=16, activation='relu')(val_5)
        val_7 = Dense(units=1, activation=None)(val_6)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv2DModel_v0B_Small_CBAM_1DConv_Share(TFModelV2):
    """
    Add CBAM (Convolutional Block Attention Module) to MyConv2DModel_v0B_Small
    Add 1x1 conv with weight share after input layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0B_Small_CBAM_1DConv_Share, self).__init__(obs_space, action_space,
                                                                        num_outputs, model_config,
                                                                        name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')
        layer_0 = Conv2D(filters=16, kernel_size=1, activation='relu')(self.inputs)

        """ Policy network """
        layer_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (layer_0)
        channel_attention_1 = ChannelAttentionModule(layer_1)
        spatial_attention_1 = SpatialAttentionModule(channel_attention_1, kernel_size=3)
        res_1 = Add()([layer_1, spatial_attention_1])
        layer_1_bn = BatchNormalization()(res_1)

        layer_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_1_bn)
        channel_attention_2 = ChannelAttentionModule(layer_2)
        spatial_attention_2 = SpatialAttentionModule(channel_attention_2, kernel_size=2)
        res_2 = Add()([layer_2, spatial_attention_2])
        layer_2_bn = BatchNormalization()(res_2)

        layer_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_2_bn)

        layer_4 = Dense(units=32, activation='relu')(layer_3)
        layer_5 = Dropout(rate=0.3)(layer_4)
        layer_6 = Dense(units=16, activation='relu')(layer_5)
        layer_out = Dense(units=num_outputs, activation=None)(layer_6)

        """ Value network """
        val_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (layer_0)
        val_channel_attention_1 = ChannelAttentionModule(val_1)
        val_spatial_attention_1 = SpatialAttentionModule(val_channel_attention_1, kernel_size=3)
        val_res_1 = Add()([val_1, val_spatial_attention_1])
        val_1_bn = BatchNormalization()(val_res_1)

        val_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_1_bn)
        val_channel_attention_2 = ChannelAttentionModule(val_2)
        val_spatial_attention_2 = SpatialAttentionModule(val_channel_attention_2, kernel_size=2)
        val_res_2 = Add()([val_2, val_spatial_attention_2])
        val_2_bn = BatchNormalization()(val_res_2)

        val_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_2_bn)

        val_4 = Dense(units=32, activation='relu')(val_3)
        val_5 = Dropout(rate=0.3)(val_4)
        val_6 = Dense(units=16, activation='relu')(val_5)
        val_7 = Dense(units=1, activation=None)(val_6)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyConv2DModel_v0B_Small_CBAM_D2RL(TFModelV2):
    """
    Add CBAM (Convolutional Block Attention Module) to MyConv2DModel_v0B_Small
    Add Skip connection from conv output to dense
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyConv2DModel_v0B_Small_CBAM_D2RL, self).__init__(obs_space, action_space, num_outputs,
                                                                model_config,
                                                                name)
        """
        obs_space.shape:(10,10,3),   action_space.n:5,   num_outputs:5
        """
        self.inputs = Input(shape=obs_space.shape, name='observation')

        """ Policy network """
        layer_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        channel_attention_1 = ChannelAttentionModule(layer_1)
        spatial_attention_1 = SpatialAttentionModule(channel_attention_1, kernel_size=3)
        res_1 = Add()([layer_1, spatial_attention_1])
        layer_1_bn = BatchNormalization()(res_1)

        layer_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_1_bn)
        channel_attention_2 = ChannelAttentionModule(layer_2)
        spatial_attention_2 = SpatialAttentionModule(channel_attention_2, kernel_size=2)
        res_2 = Add()([layer_2, spatial_attention_2])
        layer_2_bn = BatchNormalization()(res_2)

        layer_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (layer_2_bn)

        layer_4 = Dense(units=32, activation='relu')(layer_3)
        layer_4_d2rl = Concatenate(axis=-1)([layer_4, layer_3])

        layer_5 = Dropout(rate=0.3)(layer_4_d2rl)
        layer_6 = Dense(units=16, activation='relu')(layer_5)
        layer_6_d2rl = Concatenate(axis=-1)([layer_6, layer_3])

        layer_out = Dense(units=num_outputs, activation=None)(layer_6_d2rl)

        """ Value network """
        val_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='valid') \
            (self.inputs)
        val_channel_attention_1 = ChannelAttentionModule(val_1)
        val_spatial_attention_1 = SpatialAttentionModule(val_channel_attention_1, kernel_size=3)
        val_res_1 = Add()([val_1, val_spatial_attention_1])
        val_1_bn = BatchNormalization()(val_res_1)

        val_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_1_bn)
        val_channel_attention_2 = ChannelAttentionModule(val_2)
        val_spatial_attention_2 = SpatialAttentionModule(val_channel_attention_2, kernel_size=2)
        val_res_2 = Add()([val_2, val_spatial_attention_2])
        val_2_bn = BatchNormalization()(val_res_2)

        val_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid') \
            (val_2_bn)

        val_4 = Dense(units=32, activation='relu')(val_3)
        val_4_d2rl = Concatenate(axis=-1)([val_4, val_3])

        val_5 = Dropout(rate=0.3)(val_4_d2rl)
        val_6 = Dense(units=16, activation='relu')(val_5)
        val_6_d2rl = Concatenate(axis=-1)([val_6, val_3])

        val_7 = Dense(units=1, activation=None)(val_6_d2rl)
        val_out = Lambda(lambda x: tf.squeeze(x, axis=1))(val_7)

        self.base_model = Model(self.inputs, [layer_out, val_out])
        # self.base_model.summary()
        # plot_model(self.base_model, to_file='base_model.png', show_shapes=True)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class Bottleneck(Model):
    def __init__(self, growth_facotr, drop_rate):
        super(Bottleneck, self).__init__()

        self.bn1 = BatchNormalization()
        self.av1 = Activation('relu')
        self.conv1 = Conv2D(filters=4 * growth_facotr,
                            kernel_size=(1, 1),
                            strides=1,
                            padding='same')

        self.bn2 = BatchNormalization()
        self.av2 = Activation('relu')
        self.conv2 = Conv2D(filters=growth_facotr,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same')
        self.dropout = Dropout(rate=drop_rate)

        self.list_layers = [self.bn1,
                            self.av1,
                            self.conv1,
                            self.bn2,
                            self.av2,
                            self.conv2,
                            self.dropout]

    def call(self, x):
        y = x
        for layer in self.list_layers:
            y = layer(y)

        y = Concatenate(axis=-1)([x, y])
        return y


class DenseBlock(Model):
    def __init__(self, num_layers, growth_factor, drop_rate):
        super(DenseBlock, self).__init__()

        self.num_layers = num_layers
        self.growth_factor = growth_factor
        self.drop_rate = drop_rate

        self.list_layers = []
        for _ in range(num_layers):
            self.list_layers.append(Bottleneck(growth_facotr=self.growth_factor,
                                               drop_rate=self.drop_rate))

    def call(self, x):
        for layer in self.list_layers:
            x = layer(x)
        return x


class TransitionLayer(Model):
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()

        self.bn = BatchNormalization()
        self.av = Activation('relu')
        self.conv = Conv2D(filters=out_channels,
                           kernel_size=(1, 1),
                           strides=1,
                           padding='same')
        self.avgpool = AveragePooling2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same')

        self.list_layers = [self.bn,
                            self.av,
                            self.conv,
                            self.avgpool]

    def call(self, x):
        for layer in self.list_layers:
            x = layer(x)
        return x


class DenseNetModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, Model_config, name):
        super(DenseNetModel, self).__init__(obs_space, action_space, num_outputs, Model_config, name)
        num_init_features = 16
        growth_factor = 16
        block_layers = [2, 2, 2, 1]
        compression_factor = 0.5
        drop_rate = 0.2

        # First convolutin + batch normalization + pooling
        self.conv = Conv2D(filters=num_init_features,
                           kernel_size=(1, 1),
                           strides=1,
                           padding='same')

        ### Policy net
        # DenseBlock 1
        self.pol_num_channels = num_init_features
        self.pol_dense_block_1 = DenseBlock(num_layers=block_layers[0],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 1
        self.pol_num_channels += growth_factor * block_layers[0]
        self.pol_num_channels *= compression_factor
        self.pol_transition_1 = TransitionLayer(out_channels=int(self.pol_num_channels))

        # DenseBlock 2
        self.pol_dense_block_2 = DenseBlock(num_layers=block_layers[1],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 2
        self.pol_num_channels += growth_factor * block_layers[1]
        self.pol_num_channels *= compression_factor
        self.pol_transition_2 = TransitionLayer(out_channels=int(self.pol_num_channels))

        # DenseBlock 3
        self.pol_dense_block_3 = DenseBlock(num_layers=block_layers[2],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 3
        self.pol_num_channels += growth_factor * block_layers[2]
        self.pol_num_channels *= compression_factor
        self.pol_transition_3 = TransitionLayer(out_channels=int(self.pol_num_channels))

        # Dense Block 4
        self.pol_dense_block_4 = DenseBlock(num_layers=block_layers[3],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)

        # Output global_average_pooling + full_connection
        self.pol_avgpool = GlobalAveragePooling2D()
        self.pol_fc = Dense(units=5, activation=None)

        # Define policy model
        inputs = Input(shape=obs_space.shape, name='observations')
        conv1_out = self.conv(inputs)

        x = self.pol_dense_block_1(conv1_out)
        x = self.pol_transition_1(x)

        # x = self.pol_dense_block_2(x)
        # x = self.pol_transition_2(x)

        # x = self.pol_dense_block_3(x)
        # x = self.pol_transition_3(x)

        x = self.pol_dense_block_4(x)
        x = self.pol_avgpool(x)
        pol_out = self.pol_fc(x)  # (None, 5)

        ### Value net
        # DenseBlock 1
        self.val_num_channels = num_init_features
        self.val_dense_block_1 = DenseBlock(num_layers=block_layers[0],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 1
        self.val_num_channels += growth_factor * block_layers[0]
        self.val_num_channels *= compression_factor
        self.val_transition_1 = TransitionLayer(out_channels=int(self.val_num_channels))

        # DenseBlock 2
        self.val_dense_block_2 = DenseBlock(num_layers=block_layers[1],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 2
        self.val_num_channels += growth_factor * block_layers[1]
        self.val_num_channels *= compression_factor
        self.val_transition_2 = TransitionLayer(out_channels=int(self.val_num_channels))

        # DenseBlock 3
        self.val_dense_block_3 = DenseBlock(num_layers=block_layers[2],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 3
        self.val_num_channels += growth_factor * block_layers[2]
        self.val_num_channels *= compression_factor
        self.val_transition_3 = TransitionLayer(out_channels=int(self.val_num_channels))

        # Dense Block 4
        self.val_dense_block_4 = DenseBlock(num_layers=block_layers[3],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)

        # Output global_average_pooling + full_connection
        self.val_avgpool = GlobalAveragePooling2D()
        self.val_fc = Dense(units=1, activation=None)

        # Define value model
        y = self.val_dense_block_1(conv1_out)
        y = self.val_transition_1(y)

        # y = self.val_dense_block_2(y)
        # y = self.val_transition_2(y)

        # y = self.val_dense_block_3(y)
        # y = self.val_transition_3(y)

        y = self.val_dense_block_4(y)
        y = self.val_avgpool(y)
        val_out = self.val_fc(y)  # (None, 1)

        self.base_model = Model(inputs, [pol_out, val_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)  # (None,5), (None,1)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class DenseNetModelLarge(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, Model_config, name):
        super(DenseNetModelLarge, self).__init__(obs_space, action_space, num_outputs, Model_config, name)
        num_init_features = 32
        growth_factor = 32
        block_layers = [3, 2, 2, 1]
        compression_factor = 0.5
        drop_rate = 0.2

        # First convolutin + batch normalization + pooling
        self.conv = Conv2D(filters=num_init_features,
                           kernel_size=(1, 1),
                           strides=1,
                           padding='same')

        ### Policy net
        # DenseBlock 1
        self.pol_num_channels = num_init_features
        self.pol_dense_block_1 = DenseBlock(num_layers=block_layers[0],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 1
        self.pol_num_channels += growth_factor * block_layers[0]
        self.pol_num_channels *= compression_factor
        self.pol_transition_1 = TransitionLayer(out_channels=int(self.pol_num_channels))

        # DenseBlock 2
        self.pol_dense_block_2 = DenseBlock(num_layers=block_layers[1],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 2
        self.pol_num_channels += growth_factor * block_layers[1]
        self.pol_num_channels *= compression_factor
        self.pol_transition_2 = TransitionLayer(out_channels=int(self.pol_num_channels))

        # DenseBlock 3
        self.pol_dense_block_3 = DenseBlock(num_layers=block_layers[2],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 3
        self.pol_num_channels += growth_factor * block_layers[2]
        self.pol_num_channels *= compression_factor
        self.pol_transition_3 = TransitionLayer(out_channels=int(self.pol_num_channels))

        # Dense Block 4
        self.pol_dense_block_4 = DenseBlock(num_layers=block_layers[3],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)

        # Output global_average_pooling + full_connection
        self.pol_avgpool = GlobalAveragePooling2D()
        self.pol_fc = Dense(units=5, activation=None)

        # Define policy model
        inputs = Input(shape=obs_space.shape, name='observations')
        conv1_out = self.conv(inputs)

        x = self.pol_dense_block_1(conv1_out)
        x = self.pol_transition_1(x)

        x = self.pol_dense_block_2(x)
        x = self.pol_transition_2(x)

        x = self.pol_dense_block_3(x)
        x = self.pol_transition_3(x)

        x = self.pol_dense_block_4(x)
        x = self.pol_avgpool(x)
        pol_out = self.pol_fc(x)  # (None, 5)

        ### Value net
        # DenseBlock 1
        self.val_num_channels = num_init_features
        self.val_dense_block_1 = DenseBlock(num_layers=block_layers[0],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 1
        self.val_num_channels += growth_factor * block_layers[0]
        self.val_num_channels *= compression_factor
        self.val_transition_1 = TransitionLayer(out_channels=int(self.val_num_channels))

        # DenseBlock 2
        self.val_dense_block_2 = DenseBlock(num_layers=block_layers[1],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 2
        self.val_num_channels += growth_factor * block_layers[1]
        self.val_num_channels *= compression_factor
        self.val_transition_2 = TransitionLayer(out_channels=int(self.val_num_channels))

        # DenseBlock 3
        self.val_dense_block_3 = DenseBlock(num_layers=block_layers[2],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)
        # Transition Layer 3
        self.val_num_channels += growth_factor * block_layers[2]
        self.val_num_channels *= compression_factor
        self.val_transition_3 = TransitionLayer(out_channels=int(self.val_num_channels))

        # Dense Block 4
        self.val_dense_block_4 = DenseBlock(num_layers=block_layers[3],
                                            growth_factor=growth_factor,
                                            drop_rate=drop_rate)

        # Output global_average_pooling + full_connection
        self.val_avgpool = GlobalAveragePooling2D()
        self.val_fc = Dense(units=1, activation=None)

        # Define value model
        y = self.val_dense_block_1(conv1_out)
        y = self.val_transition_1(y)

        y = self.val_dense_block_2(y)
        y = self.val_transition_2(y)

        y = self.val_dense_block_3(y)
        y = self.val_transition_3(y)

        y = self.val_dense_block_4(y)
        y = self.val_avgpool(y)
        val_out = self.val_fc(y)  # (None, 1)

        self.base_model = Model(inputs, [pol_out, val_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)  # (None,5), (None,1)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class DenseNetModelLargeShare(TFModelV2):
    """
    Densenet is shared between policy and value network
    """

    def __init__(self, obs_space, action_space, num_outputs, Model_config, name):
        super(DenseNetModelLargeShare, self).__init__(obs_space, action_space, num_outputs, Model_config,
                                                      name)
        num_init_features = 32
        growth_factor = 32
        block_layers = [3, 2, 2, 1]
        compression_factor = 0.5
        drop_rate = 0.2

        # First convolutin + batch normalization + pooling
        self.conv = Conv2D(filters=num_init_features,
                           kernel_size=(1, 1),
                           strides=1,
                           padding='same')

        # DenseBlock 1
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)
        # Transition Layer 1
        self.num_channels += growth_factor * block_layers[0]
        self.num_channels *= compression_factor
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))

        # DenseBlock 2
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)
        # Transition Layer 2
        self.num_channels += growth_factor * block_layers[1]
        self.num_channels *= compression_factor
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))

        # DenseBlock 3
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)
        # Transition Layer 3
        self.num_channels += growth_factor * block_layers[2]
        self.num_channels *= compression_factor
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        # Dense Block 4
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)

        # Output global_average_pooling
        self.avgpool = GlobalAveragePooling2D()

        inputs = Input(shape=obs_space.shape, name='observations')
        conv1_out = self.conv(inputs)

        x = self.dense_block_1(conv1_out)
        x = self.transition_1(x)

        x = self.dense_block_2(x)
        x = self.transition_2(x)

        x = self.dense_block_3(x)
        x = self.transition_3(x)

        x = self.dense_block_4(x)
        x = self.avgpool(x)

        ### Policy net
        self.pol_1 = Dense(units=32, activation='relu')
        self.pol_fc = Dense(units=5, activation=None)
        y = self.pol_1(x)
        pol_out = self.pol_fc(y)  # (None, 5)

        ### Value net
        self.val_1 = Dense(units=32, activation='relu')
        self .val_fc = Dense(units=1, activation=None)
        z = self.val_1(x)
        val_out = self.val_fc(z)  # (None, 1)

        self.base_model = Model(inputs, [pol_out, val_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)  # (None,5), (None,1)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class DenseNetModelLargeShareSimple(TFModelV2):
    """
    Densenet is shared between policy and value network
    """

    def __init__(self, obs_space, action_space, num_outputs, Model_config, name):
        super(DenseNetModelLargeShareSimple, self).__init__(obs_space, action_space, num_outputs, Model_config,
                                                      name)
        num_init_features = 32
        growth_factor = 32
        block_layers = [3, 2, 2, 1]
        compression_factor = 0.5
        drop_rate = 0.2

        # First convolutin + batch normalization + pooling
        self.conv = Conv2D(filters=num_init_features,
                           kernel_size=(1, 1),
                           strides=1,
                           padding='same')

        # DenseBlock 1
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)
        # Transition Layer 1
        self.num_channels += growth_factor * block_layers[0]
        self.num_channels *= compression_factor
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))

        # DenseBlock 2
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)
        # Transition Layer 2
        self.num_channels += growth_factor * block_layers[1]
        self.num_channels *= compression_factor
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))

        # DenseBlock 3
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)
        # Transition Layer 3
        self.num_channels += growth_factor * block_layers[2]
        self.num_channels *= compression_factor
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        # Dense Block 4
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3],
                                        growth_factor=growth_factor,
                                        drop_rate=drop_rate)

        # Output global_average_pooling
        self.avgpool = GlobalAveragePooling2D()

        inputs = Input(shape=obs_space.shape, name='observations')
        conv1_out = self.conv(inputs)

        x = self.dense_block_1(conv1_out)
        x = self.transition_1(x)

        x = self.dense_block_2(x)
        x = self.transition_2(x)

        x = self.dense_block_3(x)
        x = self.transition_3(x)

        x = self.dense_block_4(x)
        x = self.avgpool(x)

        ### Policy net
        self.pol_fc = Dense(units=5, activation=None)
        pol_out = self.pol_fc(x)  # (None, 5)

        ### Value net
        self .val_fc = Dense(units=1, activation=None)
        val_out = self.val_fc(x)  # (None, 1)

        self.base_model = Model(inputs, [pol_out, val_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._value_out = self.base_model(obs)  # (None,5), (None,1)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyRNNConv1DModel_v3(RecurrentNetwork):
    """
    Not yet implemented !!!!!
    下記情報を元にして、最新の rllib version に合うように一部修正。
    とにかく「動く版」を作成するのが目的
        Information: https://github.com/ray-project/ray/issues/6928

        This was (given that you are not using tf-eager) a problem in your model.
        Here is a working version of it (see code below).
        The trick was to correctly fold the time-rank into the batch rank before pushing it through the CNN,
        then correctly unfolding it again before the LSTM pass.

        For the eager case, there is actually a bug in RLlib, which I'll fix now (issue #6732).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyRNNConv1DModel_v3, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        cnn_shape = obs_space.shape
        self.cell_size = 128

        visual_size = cnn_shape[0] * cnn_shape[1]
        state_in_h = Input(shape=(self.cell_size,), name='h')
        state_in_c = Input(shape=(self.cell_size,), name='c')
        seq_in = Input(shape=(), name='seq_in', dtype=tf.int32)

        inputs = Input(shape=(None, visual_size), name='visual_inputs')  # Add time dim
        input_visual = inputs
        input_visual = tf.reshape(input_visual, [-1, cnn_shape[0], cnn_shape[1]])

        layer_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (input_visual)
        layer_bn = BatchNormalization()(layer_1)
        layer_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn)
        layer_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (layer_2)
        layer_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_3)
        # vision_out = Flatten()(layer_3)
        vision_out = Lambda(lambda x: tf.squeeze(x, axis=1))(layer_4)

        vision_out = tf.reshape(vision_out, [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])

        lstm_out, state_h, state_c = LSTM(units=self.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          return_state=True,
                                          return_sequences=True,
                                          name='lstm')(inputs=vision_out,
                                                       mask=tf.sequence_mask(lengths=seq_in),
                                                       initial_state=[state_in_h, state_in_c])
        layer_5 = Dense(units=64, activation='relu')(lstm_out)
        layer_6 = Dropout(rate=0.3)(layer_5)
        layer_7 = Dense(units=32, activation='relu')(layer_6)
        logits = Dense(units=num_outputs, activation=None, name='logits')(layer_7)

        val_5 = Dense(units=64, activation='relu')(lstm_out)
        val_6 = Dropout(rate=0.3)(val_5)
        val_7 = Dense(units=32, activation='relu')(val_6)
        values = Dense(units=1, activation=None, name='values')(val_7)

        self.rnn_model = Model(inputs=[inputs, seq_in, state_in_h, state_in_c],
                               outputs=[logits, values, state_h, state_c])

        # self.rnn_model.summary()
        # plot_model(self.rnn_model, to_file='rnn_model.png', show_shapes=True)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # inputs:(None, None, 30),  state=[(None,128),(None,128)], seq_len:(None,)
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyRNNConv1DModel_v3A(RecurrentNetwork):
    """
    Not yet implemented !!!!!
    重み共有すると、FWDですら上手く行かないので、LSTMだけ共有するArchitectureとした
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyRNNConv1DModel_v3A, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        cnn_shape = obs_space.shape
        self.cell_size = 256

        visual_size = cnn_shape[0] * cnn_shape[1]
        state_in_h = Input(shape=(self.cell_size,), name='h')
        state_in_c = Input(shape=(self.cell_size,), name='c')
        seq_in = Input(shape=(), name='seq_in', dtype=tf.int32)

        inputs = Input(shape=(None, visual_size), name='visual_inputs')  # Add time dim
        input_visual = inputs
        input_visual = tf.reshape(input_visual, [-1, cnn_shape[0], cnn_shape[1]])

        # Policy network
        layer_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (input_visual)
        layer_bn_1 = BatchNormalization()(layer_1)

        layer_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn_1)
        layer_bn_2 = BatchNormalization()(layer_2)

        layer_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (layer_bn_2)
        layer_bn_3 = BatchNormalization()(layer_3)

        layer_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn_3)

        """ Value network """
        val_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (input_visual)
        val_bn_1 = BatchNormalization()(val_1)

        val_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (val_bn_1)
        val_bn_2 = BatchNormalization()(val_2)

        val_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (val_bn_2)
        val_bn_3 = BatchNormalization()(val_3)

        val_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (val_bn_3)

        # Concatenate convolution
        layer_out = Concatenate(axis=-1)([layer_4, val_4])
        vision_out = Lambda(lambda x: tf.squeeze(x, axis=1))(layer_out)
        vision_out = tf.reshape(vision_out, [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])

        lstm_out, state_h, state_c = LSTM(units=self.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          return_state=True,
                                          return_sequences=True,
                                          name='lstm')(inputs=vision_out,
                                                       mask=tf.sequence_mask(lengths=seq_in),
                                                       initial_state=[state_in_h, state_in_c])
        # Policy network
        layer_5 = Dense(units=128, activation='relu')(lstm_out)
        layer_6 = Dense(units=64, activation='relu')(layer_5)
        layer_7 = Dropout(rate=0.3)(layer_6)
        logits = Dense(units=num_outputs, activation=None, name='logits')(layer_7)

        # Value network
        val_5 = Dense(units=128, activation='relu')(lstm_out)
        val_6 = Dense(units=64, activation='relu')(val_5)
        val_7 = Dropout(rate=0.3)(val_6)
        values = Dense(units=1, activation=None, name='values')(val_7)

        self.rnn_model = Model(inputs=[inputs, seq_in, state_in_h, state_in_c],
                               outputs=[logits, values, state_h, state_c])

        # self.rnn_model.summary()
        # plot_model(self.rnn_model, to_file='rnn_model.png', show_shapes=True)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # inputs:(None, None, 30),  state=[(None,128),(None,128)], seq_len:(None,)
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyRNNConv1DModel_v3_Attention(RecurrentNetwork):
    """
    Not yet implemented !!!!!
    Batch_Normalization, Dropout, Block Convolutional Attention Module 追加晩
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyRNNConv1DModel_v3, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # cnn_shape = (10, 3)  # temporal setting
        cnn_shape = obs_space.shape
        self.cell_size = 128

        visual_size = cnn_shape[0] * cnn_shape[1]
        state_in_h = Input(shape=(self.cell_size,), name='h')
        state_in_c = Input(shape=(self.cell_size,), name='c')
        seq_in = Input(shape=(), name='seq_in', dtype=tf.int32)

        inputs = Input(shape=(None, visual_size), name='visual_inputs')  # Add time dim
        input_visual = inputs
        input_visual = tf.reshape(input_visual, [-1, cnn_shape[0], cnn_shape[1]])

        layer_1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid') \
            (input_visual)
        channel_attention_1 = ChannelAttentionModule(layer_1)
        spatial_attention_1 = SpatialAttentionModule(channel_attention_1, kernel_size=46)
        res_1 = Add()([layer_1, spatial_attention_1])

        layer_bn = BatchNormalization()(res_1)
        layer_2 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (layer_bn)
        channel_attention_2 = ChannelAttentionModule(layer_2)
        spatial_attention_2 = SpatialAttentionModule(channel_attention_2, kernel_size=21)
        res_2 = Add()([layer_2, spatial_attention_2])

        layer_3 = Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid') \
            (res_2)
        channel_attention_3 = ChannelAttentionModule(layer_3)
        spatial_attention_3 = SpatialAttentionModule(channel_attention_3, kernel_size=6)
        res_3 = Add()([layer_3, spatial_attention_3])

        layer_4 = Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='valid') \
            (res_3)
        # vision_out = Flatten()(layer_3)
        vision_out = Lambda(lambda x: tf.squeeze(x, axis=1))(layer_4)

        vision_out = tf.reshape(vision_out, [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])

        lstm_out, state_h, state_c = LSTM(units=self.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          return_state=True,
                                          return_sequences=True,
                                          name='lstm')(inputs=vision_out,
                                                       mask=tf.sequence_mask(lengths=seq_in),
                                                       initial_state=[state_in_h, state_in_c])
        layer_5 = Dense(units=64, activation='relu')(lstm_out)
        layer_6 = Dropout(rate=0.3)(layer_5)
        layer_7 = Dense(units=32, activation='relu')(layer_6)
        logits = Dense(units=num_outputs, activation=None, name='logits')(layer_7)

        val_5 = Dense(units=64, activation='relu')(lstm_out)
        val_6 = Dropout(rate=0.3)(val_5)
        val_7 = Dense(units=32, activation='relu')(val_6)
        values = Dense(units=1, activation=None, name='values')(val_7)

        self.rnn_model = Model(inputs=[inputs, seq_in, state_in_h, state_in_c],
                               outputs=[logits, values, state_h, state_c])

        # self.rnn_model.summary()
        # plot_model(self.rnn_model, to_file='rnn_model.png', show_shapes=True)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # inputs:(None, None, 30),  state=[(None,128),(None,128)], seq_len:(None,)
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
