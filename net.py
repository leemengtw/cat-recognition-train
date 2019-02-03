import tensorflow as tf


# chs in argument: avoid redundant variables in tf graph
#                  should do more in graph building phase; produces less garbage
# using pytorch batch_norm defaults
def batch_norm(
        x, chs, is_training, module_cnt, inference_only, eps=1e-5, momentum=.9,
        init_gamma=None, init_beta=None, init_r_mean=None, init_r_var=None):  # load from numpy
    with tf.variable_scope("bn_%02d" % module_cnt):
        gamma = tf.get_variable(
            "gamma", dtype=tf.float32, initializer=init_gamma if init_gamma else tf.ones([chs]))
        beta = tf.get_variable(
            "beta", dtype=tf.float32, initializer=init_beta if init_beta else tf.zeros([chs]))
        r_mean = tf.get_variable(
            "r_mean", dtype=tf.float32,
            initializer=init_r_mean if init_r_mean else tf.zeros([chs]), trainable=False)
        r_var = tf.get_variable(
            "r_var", dtype=tf.float32,
            initializer=init_r_var if init_r_var else tf.ones([chs]), trainable=False)
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


def conv(
        x, in_chs, out_chs, k_size, stride, module_cnt, bias, pad="SAME",
        init_weight=None, init_b=None):  # load from numpy
    with tf.variable_scope("conv_%02d" % module_cnt):
        weight = tf.get_variable(
            "kernel", [k_size, k_size, in_chs, out_chs], tf.float32,
            init_weight if init_weight else tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, weight, [1, stride, stride, 1], pad)
        if bias:
            b = tf.get_variable(
                "bias", [out_chs], tf.float32,
                init_b if init_b else tf.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        return x


def dwise_conv(
        x, in_chs, k_size, stride, module_cnt, bias, chs_mult=1, pad="SAME",
        init_weight=None, init_b=None):
    with tf.variable_scope("dwise_conv_%02d" % module_cnt):
        weight = tf.get_variable(
            "kernel", [k_size, k_size, in_chs, chs_mult], tf.float32,
            init_weight if init_weight else tf.contrib.layers.xavier_initializer())
        x = tf.nn.depthwise_conv2d(x, weight, [1, stride, stride, 1], pad)
        if bias:
            b = tf.get_variable(
                "bias", [int(in_chs * chs_mult)], tf.float32,
                init_b if init_b else tf.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        return x


def channel_shuffle(x, groups, module_cnt):
    with tf.variable_scope("channel_shuffle_%02d" % module_cnt):
        _, h, w, c = x.shape.as_list()
        x = tf.reshape(x, [-1, h, w, groups, c // groups])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [-1, h, w, c])
        return x


def shufflenet_unit(
        x, in_c, is_training, module_cnt,
        inference_only, out_c=None, init_params=None):
    with tf.variable_scope("shufflenet_unit_%02d" % module_cnt):
        if out_c:  # Downsample and double (or more) the channels
            assert out_c >= in_c * 2
            chs = out_c // 2
            x1, x2 = x, x
            # 1st branch
            x1 = dwise_conv(
                x1, in_c, 3, 2, 0, False, 1, "SAME",
                init_params[0] if init_params else None)
            x1 = batch_norm(
                x1, in_c, is_training, 1, inference_only, 1e-5, .9,
                *init_params[1:5] if init_params else None)
            x1 = conv(
                x1, in_c, chs, 1, 1, 2, False, "SAME",
                init_params[5] if init_params else None)
            x1 = batch_norm(
                x1, chs, is_training, 3, inference_only, 1e-5, .9,
                *init_params[6:10] if init_params else [None])
            # 2nd branch
            x2 = conv(
                x2, in_c, chs, 1, 1, 4, False, "SAME",
                init_params[10] if init_params else None)
            x2 = tf.nn.relu(batch_norm(
                x2, chs, is_training, 5, inference_only, 1e-5, .9,
                *init_params[11:15] if init_params else [None]))
            x2 = dwise_conv(
                x2, chs, 3, 2, 6, False, 1, "SAME",
                init_params[15] if init_params else None)
            x2 = batch_norm(
                x2, chs, is_training, 7, inference_only, 1e-5, .9,
                *init_params[16:20] if init_params else None)
            x2 = conv(
                x2, chs, chs, 1, 1, 8, False, "SAME",
                init_params[20] if init_params else None)
            x2 = batch_norm(
                x2, chs, is_training, 9, inference_only, 1e-5, .9,
                *init_params[21:25] if init_params else None)
            x = tf.nn.relu(tf.concat([x1, x2], 3))
        else:
            assert in_c % 2 == 0
            chs = in_c // 2
            left, right = x[..., :chs], x[..., chs:]
            right = conv(
                right, chs, chs, 1, 1, 0, False, "SAME",
                init_params[0] if init_params else None)  # no bias
            right = tf.nn.relu(batch_norm(
                right, chs, is_training, 1, inference_only, 1e-5, .9,
                *init_params[1:5] if init_params else [None]))
            right = dwise_conv(
                right, chs, 3, 1, 2, False, 1, "SAME",
                init_params[5] if init_params else None)
            right = batch_norm(
                right, chs, is_training, 3, inference_only, 1e-5, .9,
                *init_params[6:10] if init_params else [None])
            right = conv(
                right, chs, chs, 1, 1, 3, False, "SAME",
                init_params[10] if init_params else None)  # no bias
            right = tf.nn.relu(batch_norm(
                right, chs, is_training, 4, inference_only, 1e-5, .9,
                *init_params[11:15] if init_params else [None]))
            x = tf.concat([left, right], 3)
        return channel_shuffle(x, 2, 10 if out_c else 5)


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
    import numpy as np
    shape = [5, 5, 4]
    a = tf.placeholder(tf.float32, shape=[None, *shape])
    is_training = tf.placeholder(tf.bool)
    ab = shufflenet_unit(a, shape[-1], is_training, 0, False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(ab, feed_dict={a: np.random.rand(2, *shape), is_training: True}))


if __name__ == '__main__':
    _test()
