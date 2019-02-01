import tensorflow as tf


# chs in argument: avoid redundant variables in tf graph
#                  should do more in graph building phase; produces less garbage
# using pytorch batch_norm defaults
def batch_norm_2d(x, chs, is_training, module_cnt, inference_only, eps=1e-5, momentum=.9):
    with tf.variable_scope("bn_%02d" % module_cnt):
        gamma = tf.get_variable("gamma", [chs], tf.float32, tf.ones([chs]))
        beta = tf.get_variable("beta", [chs], tf.float32, tf.zeros([chs]))
        r_mean = tf.get_variable(
            "r_mean", [chs], tf.float32, tf.zeros([chs]), trainable=False)
        r_var = tf.get_variable(
            "r_var", [chs], tf.float32, tf.ones([chs]), trainable=False)
        # Avoid producing dirty graph
        if inference_only:
            x = tf.nn.batch_normalization(x, r_mean, r_var, beta, gamma, eps)
        else:
            def _train():
                mean, variance = tf.nn.moments(x, [0, 1, 2], name="moments")
                # not using tf.train.ExponentialMovingAverage for better variable control
                # so we can load trained variables into inference_only graph
                update_mean_op = tf.assign(
                    r_mean, r_mean * momentum + mean * (1 - momentum))
                update_var_op = tf.assign(
                    r_var, r_var * momentum + variance * (1 - momentum))
                with tf.control_dependencies([update_mean_op, update_var_op]):
                    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            x = tf.cond(
                is_training,
                _train,
                lambda: tf.nn.batch_normalization(x, r_mean, r_var, beta, gamma, eps))
        return x


def conv_2d(x, in_chs, out_chs, k_size, stride, module_cnt, bias, pad="SAME"):
    with tf.variable_scope("conv_%02d" % module_cnt):
        weight = tf.get_variable(
            "kernel", [*k_size, in_chs, out_chs], tf.float32,
            tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, weight, [1, *stride, 1], pad)
        if bias:
            b = tf.get_variable(
                "bias", [out_chs], tf.float32, tf.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        return x


def dwise_conv_2d(x, in_chs, k_size, stride, module_cnt, bias, chs_mult=1, pad="Same"):
    with tf.variable_scope("dwise_conv_%02d" % module_cnt):
        weight = tf.get_variable(
            "kernel", [*k_size, in_chs, chs_mult], tf.float32,
            tf.contrib.layers.xavier_initializer())
        x = tf.nn.depthwise_conv2d(x, weight, [1, *stride, 1], pad)
        if bias:
            b = tf.get_variable(
                "bias", [int(in_chs * chs_mult)], tf.float32, tf.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        return x


def channel_shuffle(x, groups, module_cnt):
    with tf.variable_scope("channel_shuffle_%02d" % module_cnt):
        n, h, w, c = x.shape.as_list()
        x = tf.reshape(x, [n, h, w, groups, c // groups])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [n, h, w, c])
        return x


class Net():

    def __init__(self, x, cls=2, alpha=1., input_size=(224, 224), inference_only=False):
        assert len(input_size) == 2
        self.output_neuron = cls if cls > 2 else 1
        self.x = x
        self.inference_only = inference_only
        # inference_only: so there's no need to feed is_training placholder on frozen graph
        #                 and hope graph optimization tool will fuse the constants.
        self.is_training = tf.constant(True) if inference_only else\
            tf.placeholder(tf.bool, name="is_training")
        self.input_size = (None, *input_size, 3)
        if alpha == 0.5:
            self.first_block_chs = 48
        elif alpha == 1.:
            self.first_block_chs = 116
        elif alpha == 1.5:
            self.first_block_chs = 176
        elif alpha == 2.:
            self.first_block_chs = 244
        else:
            import logging
            logging.error("Unexpected alpha, which should be 0.5, 1.0, 1.5, or 2.0")
            raise ValueError
        self.build_net()

    def build_net(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def __call__(self, x):
        pass


def _test():
    pass


if __name__ == '__main__':
    _test()
