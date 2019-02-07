import os
import logging
import pickle
import numpy as np
import torch
import tensorflow as tf
from net import Net
from dataset import Dataset


def reuse_test():
    logging.warning("Test functions do tf.reset_default_graph()!")
    tf.reset_default_graph()

    shape = [2, 224, 224, 3]
    nx = (np.random.rand(*shape) * 10 - 5).astype(np.float32)
    a = tf.placeholder(tf.float32, shape=[None, *shape[1:]])
    tnet = Net(a)
    ab1 = tnet.out
    vnet = Net(a, reuse=True)
    ab2 = vnet.out
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z1 = sess.run(ab1, {a: nx, tnet.is_training: False})
        z2 = sess.run(ab2, {a: nx, vnet.is_training: False})
    diff = np.sqrt(np.mean((z1-z2)**2))
    logging.info("Diff between original net and reused net: %f" % diff)
    return diff == 0.


def convert_pytorch_weight_test(width_mult, path):

    import logging
    from shufflenet_v2_pytorch.shufflenetv2_base import shufflenetv2_base
    logging.warning("Test functions do tf.reset_default_graph()!")
    tf.reset_default_graph()

    def p_load(net, sd):
        cnt = 0
        net_keys = [
            k for k in list(net.state_dict().keys()) if not k.endswith("num_batches_tracked")]
        for from_key, to_key in zip(sd.keys(), net_keys):
            net.state_dict()[to_key].copy_(sd[from_key])
            cnt += 1
        logging.info("%d params loaded" % cnt)

    ar = np.random.rand(2, 224, 224, 3).astype(np.float32) * 10 - 5
    tnet = shufflenetv2_base(width_mult)
    tar = torch.from_numpy(ar.transpose(0, 3, 1, 2))
    # z = torch.load(os.path.join(
    #     "shufflenet_v2_pytorch", "shufflenetv2_x1_69.402_88.374.pth.tar"))
    z = torch.load(path)
    # tnet.load_state_dict(z, strict=False)
    with torch.no_grad():
        p_load(tnet, z)
    params = []
    for k, v in z.items():
        v = v.numpy()
        assert v.ndim in [1, 2, 4], (k, v.ndim)
        if v.ndim == 4:
            if v.shape[1] == 1:
                v = v.transpose(2, 3, 0, 1)
            else:
                v = v.transpose(2, 3, 1, 0)
        params.append(v)
    inference_only = False
    is_training = True
    shape = [224, 224]
    a = tf.placeholder(tf.float32, shape=[None, *shape, 3])
    net = Net(a, alpha=width_mult, inference_only=inference_only,
              init_params=params, test_convert=True)
    ab = net.out
    if inference_only or not is_training:
        tnet = tnet.eval()
    else:
        tnet = tnet.train()
    with torch.no_grad():
        tout = tnet(tar).numpy().transpose(0, 2, 3, 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z = sess.run(ab, {a: ar} if inference_only else {a: ar, net.is_training: is_training})
    diff0 = np.sqrt(np.mean((tout-z)**2))
    logging.info("Diff between tensorflow and pytorch: %f" % diff0)
    if diff0 > 1e-5:
        logging.warning(
            "Diff between tensorflow and pytorch is bigger than 1e-5 (%f)" % diff0)
    param_path = "imagenet_pretrained_shufflenetv2_%.1f.pkl" % width_mult
    with open(param_path, "wb") as f:
        pickle.dump(params, f)
    logging.info("Transposed numpy weights from pytorch model saved to %s" % param_path)
    del net
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, shape=[None, *shape, 3])
    net = Net(a, alpha=width_mult, inference_only=inference_only,
              init_params=params, test_convert=True)
    ab = net.out
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load_from_numpy(sess, params)
        zz = sess.run(ab, {a: ar} if inference_only else {a: ar, net.is_training: is_training})
    diff1 = np.sqrt(np.mean((zz-z)**2))
    logging.info("Diff between load_from_numpy and list of params: %f" % diff1)
    del net
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, shape=[None, *shape, 3])
    net = Net(a, alpha=width_mult, inference_only=inference_only,
              init_params=params, test_convert=True)
    ab = net.out
    tmp_file = "tmp.pkl"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load_from_numpy(sess, path=param_path)
        zzz = sess.run(ab, {a: ar} if inference_only else {a: ar, net.is_training: is_training})
        net.save_to_numpy(sess, tmp_file)
    diff2 = np.sqrt(np.mean((zzz-z)**2))
    logging.info("Diff between load_from_numpy and path: %f" % diff2)
    diff3 = np.sqrt(np.mean((zzz-zz)**2))
    del net
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, shape=[None, *shape, 3])
    net = Net(a, alpha=width_mult, inference_only=inference_only,
              init_params=params, test_convert=True)
    ab = net.out
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load_from_numpy(sess, path=tmp_file)
        os.remove(tmp_file)
        zzzz = sess.run(ab, {a: ar} if inference_only else {a: ar, net.is_training: is_training})
    diff4 = np.sqrt(np.mean((zzzz-z)**2))
    logging.info("Diff between save_to_numpy and load_from_numpy: %f" % diff4)
    return diff0 < 1e-5 and diff1 == 0. and diff2 == 0. and diff3 == 0. and diff4 == 0.


def tf_saver_test():
    # Check if inference_only mode works
    logging.warning("Test functions do tf.reset_default_graph()!")
    tf.reset_default_graph()
    shape = [2, 224, 224, 3]
    directory = "ckpts"
    a = tf.placeholder(tf.float32, shape=[None, *shape[1:]])
    nx = (np.random.rand(*shape) * 10 - 5).astype(np.float32)
    net = Net(a, inference_only=False)
    ab = net.out
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("runs", sess.graph)
        sess.run(tf.global_variables_initializer())
        z = sess.run(ab, feed_dict={a: nx, net.is_training: False})
        net.save(sess, directory, "0")
        writer.close()
    del net
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, shape=[None, *shape[1:]])
    net = Net(a, inference_only=True)
    vnet = Net(a, inference_only=True, reuse=True)
    ab = net.out
    vab = vnet.out
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load(sess, directory)
        zz = sess.run(ab, feed_dict={a: nx})
        zzz = sess.run(vab, feed_dict={a: nx})
    diff0 = np.sqrt(np.mean((z-zz)**2))
    diff1 = np.sqrt(np.mean((zz-zzz)**2))
    logging.info("Diff between original params and loaded params: %f" % diff0)
    logging.info("Diff between loaded params and reused net: %f" % diff1)
    return diff0 == 0. and diff1 == 0.


def dataset_test():
    epochs = 2
    is_train = True
    is_val = False
    d = Dataset("datasets", train=is_train, debug=True)
    sess = tf.Session()
    next_item = d.get_next()
    not_gone_through = False
    for e in range(1, epochs + 1):
        d.initialize(sess, not is_val)
        for i in range(1, len(d) + 1):
            v, k = sess.run(next_item)
    try:
        while True:
            sess.run(next_item)  # Should raise exception
            not_gone_through = True  # Should not run
    except tf.errors.OutOfRangeError:
        pass
    return not not_gone_through


def main():
    oks = []
    original_lvl = logging.getLogger().getEffectiveLevel()
    logging.basicConfig(level=logging.INFO)
    logging.info("Checking dataset...")
    oks.append(dataset_test())
    logging.info("OK" if oks[-1] else "Failed")
    logging.info("Checking convert and load_from_numpy functionality; it may take a while...")
    oks.append(convert_pytorch_weight_test(1.0, os.path.join(
        "shufflenet_v2_pytorch", "shufflenetv2_x1_69.402_88.374.pth.tar")))
    logging.info("OK" if oks[-1] else "Failed")
    logging.info("Checking convert and load_from_numpy functionality; it may take a while...")
    oks.append(convert_pytorch_weight_test(0.5, os.path.join(
        "shufflenet_v2_pytorch", "shufflenetv2_x0.5_60.646_81.696.pth.tar")))
    logging.info("OK" if oks[-1] else "Failed")
    logging.info("Checking save/load functionality; it may take a while...")
    oks.append(tf_saver_test())
    logging.info("OK" if oks[-1] else "Failed")
    logging.info("Checking reuse functionality; it may take a while...")
    oks.append(reuse_test())
    logging.info("OK" if oks[-1] else "Failed")
    logging.basicConfig(level=original_lvl)
    assert all(oks)


if __name__ == '__main__':
    main()
