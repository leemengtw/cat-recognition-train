import os
import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from dataset import Dataset, MEAN, STD
from net import Net
from tensorflow.tools.graph_transforms import TransformGraph

__no_tqdm__ = False
try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    __no_tqdm__ = True


def _tqdm(res, *args, **kwargs):
    return res


__optimizers__ = {
    "adadelta": lambda lr, arg1, arg2, arg3=None: tf.train.AdadeltaOptimizer(lr, arg1, arg2),
    "adagrad": lambda lr, arg1, arg2=None, arg3=None: tf.train.AdagradOptimizer(lr, arg1),
    "adam": lambda lr, arg1, arg2, arg3: tf.train.AdamOptimizer(lr, arg1, arg2, arg3),
    "nadam": lambda lr, arg1, arg2, arg3: tf.contrib.opt.NadamOptimizer(lr, arg1, arg2, arg3),
    "rmsprop": lambda lr, arg1, arg2, arg3: tf.train.RMSPropOptimizer(lr, arg1, arg2, arg3),
    "sgd": lambda lr, arg1, arg2, arg3=None: tf.train.MomentumOptimizer(
                                             lr, arg1, use_nesterov=arg3)
}


class Trainer():

    def __init__(
            self,
            data_folder="datasets",
            batch_size=64,
            input_size=224,
            valset_ratio=.2,
            epochs=60,
            alpha=0.5,
            optim="adam",
            init_lr=1e-3,
            optim_args=[.9, .999, 1e-8],
            lr_decay_step=20,
            lr_decay_rate=.1,
            init_param=None,
            logdir="runs",
            savedir="ckpts",
            random_seed=0,
            logger=None,
            show_progress=True,
            restore=None,
            debug=False):
        self.ascii = os.name == "nt"
        self.debug = debug
        if optim == "sgd":
            optim_args[2] = optim_args[2] >= 1.
        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.info)
            self.logger.warning("You are using the root logger, which has bad a format.")
            self.logger.warning("Please consider passing a better logger.")
        self.best_acc, self.best_avg_val_loss = 0., float("inf")
        self.epoch = 1
        self.alpha = alpha
        subdir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.logdir = os.path.join(logdir, subdir)
        self.savedir = os.path.join(savedir, subdir)
        if restore is not None:
            self.logger.info("Restoring training progress from %s..." % restore)
            with open(os.path.join(restore, "config.pkl"), "rb") as f:
                (data_folder, batch_size, input_size, valset_ratio, epochs, self.alpha,
                 optim, init_lr, optim_args, lr_decay_step, lr_decay_rate, self.logdir,
                 self.savedir, random_seed) = pickle.load(f)
            with open(os.path.join(restore, "current_status.txt"), "r") as f:
                self.epoch, self.best_acc, self.best_avg_val_loss = f.read().splitlines()
                self.epoch, self.best_acc, self.best_avg_val_loss = \
                    int(self.epoch), float(self.best_acc), float(self.best_avg_val_loss)
            if os.path.exists(os.path.join(restore, "net_latest.meta")) and\
                    (init_param is not None):
                self.logger.warning(
                    "Checkpoint in %s exists; not initializing params from %s" % (
                        restore, init_param))
                init_param = None
        else:
            if not os.path.isdir(self.savedir):
                os.makedirs(self.savedir)
            with open(os.path.join(savedir, "history"), "a") as f:
                f.write("%s\n" % subdir)
            with open(os.path.join(self.savedir, "config.pkl"), "wb") as f:
                pickle.dump(
                    (data_folder, batch_size, input_size, valset_ratio, epochs, self.alpha,
                     optim, init_lr, optim_args, lr_decay_step, lr_decay_rate, self.logdir,
                     self.savedir, random_seed),
                    f)
            with open(os.path.join(self.savedir, "current_status.txt"), "w") as f:
                f.write("%d\n%f\n%f" % (self.epoch, self.best_acc, self.best_avg_val_loss))
        if not show_progress or __no_tqdm__:
            self.tqdm = _tqdm
        else:
            self.tqdm = tqdm
        if epochs > lr_decay_step:
            lr_bnds = [i for i in range(lr_decay_step, epochs, lr_decay_step)]
        else:
            self.logger.warning("lr_decay_step > epochs; lr decay will not be performed.")
            lr_bnds = []
        lr_vals = [init_lr * lr_decay_rate ** i for i in range(len(lr_bnds) + 1)]
        self.logger.debug("LR Boundarys: %s, LR Vals: %s" % (lr_bnds, lr_vals))
        tf.set_random_seed(random_seed)
        self.epochs = epochs
        self.logger.info("Model checkpoints will be saved to %s" % self.savedir)
        self.logger.info("Summary for TensorBoard will be saved to %s" % self.logdir)
        self.logger.info(
            "You can use \"tensorboard --logdir %s\" to see all training summaries." % logdir)
        # logdir: folder containing all training histories
        # self.logdir: folder containing current training summary
        self.logger.info("Preparing dataset...")
        self.trainset = Dataset(
            data_folder, True, (input_size, input_size),
            batch_size, None, valset_ratio, random_seed, debug)
        self.input_size = input_size
        self.logger.debug("%d training instances, %d validation instances" % (
            self.trainset.train_length, self.trainset.val_length))
        self.total_pred = tf.constant(self.trainset.val_length, name="total_pred")
        self.logger.info("Generating training operations...")
        x, y = self.trainset.get_next()
        self._build_train_graph(
            x, y, optim, lr_bnds, lr_vals, optim_args, init_param)
        self.logger.info("Generating validation operations...")
        self._build_val_graph(x, y)
        self.sess = tf.Session()
        self.sum_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if restore is not None:
            if os.path.exists(os.path.join(restore, "net_latest.meta")):
                self.train_net.load(self.sess, restore, "net_latest")
                self.logger.info("Network params restored from %s" % restore)
            else:
                self.logger.warning(
                    "No network ckpt file found in %s; not loading net params" % restore)
            if os.path.exists(os.path.join(restore, "optim_latest.meta")):
                self.optim_saver.restore(self.sess, os.path.join(restore, "optim_latest"))
                self.logger.info("Optimizer gradients restored from %s" % restore)
            else:
                self.logger.warning(
                    "No optimizer ckpt file found in %s; not loading optim params" % restore)

    def _build_train_graph(
            self, x, y, optim, bnds, vals, optim_args, init_param=None):
        self.train_net = Net(
            x, alpha=self.alpha, input_size=(self.input_size, self.input_size),
            init_param=init_param)
        loss, self.train_accum_loss, self.train_avg_loss, self.train_reset_loss = \
            self._build_loss_graph(self.train_net.out, y, "train")
        self.global_step = tf.placeholder(tf.int32)
        if len(bnds) > 0:
            lr = tf.train.piecewise_constant(self.global_step, bnds, vals)
        else:
            lr = vals[0]
        with tf.variable_scope("optim_vars"):
            self.train_op = __optimizers__[optim](lr, *optim_args).minimize(loss)
        self.optim_vars = [
            v for v in tf.global_variables() if v.name.startswith("optim_vars")]
        self.optim_saver = tf.train.Saver(self.optim_vars)
        self.train_summary = tf.summary.scalar("training loss", self.train_avg_loss)

    def _build_val_graph(self, x, y):
        self.val_net = Net(
            x, alpha=self.alpha, input_size=(self.input_size, self.input_size), reuse=True)
        cur_currect = tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(self.val_net.out > 0, tf.int32), y), tf.int32))
        total_correct = tf.get_variable("total_correct", None, tf.int32, tf.constant(0))
        self.accum_correct = tf.assign(total_correct, total_correct + cur_currect)
        self.reset_correct = tf.assign(total_correct, 0)
        self.accuracy = total_correct / self.total_pred
        _, self.val_accum_loss, self.val_avg_loss, self.val_reset_loss = \
            self._build_loss_graph(self.val_net.out, y, "val")
        self.val_summary = tf.summary.merge([
            tf.summary.scalar("validation loss", self.val_avg_loss),
            tf.summary.scalar("validation accuracy", self.accuracy)])

    def _build_loss_graph(self, y_pred, y_label, mode="train"):
        assert mode in ("train", "val")
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(y_label, tf.float32), logits=y_pred)
        loss_sum = tf.reduce_sum(loss)
        loss_mean = tf.reduce_mean(loss)
        total_loss = tf.get_variable(
            "total_loss_%s" % mode, None, tf.float32, tf.constant(0.))
        accum_loss = tf.assign(total_loss, total_loss + loss_sum)
        avg_loss = total_loss / tf.cast(self.total_pred, tf.float32)
        reset_loss = tf.assign(total_loss, 0.)
        return loss_mean, accum_loss, avg_loss, reset_loss

    def eval(self, epoch, save=True):
        self.trainset.initialize(self.sess, False)  # inits valset inside trainset
        for _ in self.tqdm(range(len(self.trainset)),
                           desc="[Epoch %d Evaluation]" % epoch, ascii=self.ascii):
            self.sess.run(
                [self.accum_correct, self.val_accum_loss], {self.val_net.is_training: False})
        acc, loss, summary = self.sess.run([self.accuracy, self.val_avg_loss, self.val_summary])
        self.logger.info(
            "Epoch %d evaluation done, acc: %f, avg_val_loss: %f" % (epoch, acc, loss))
        self.sum_writer.add_summary(summary, epoch)
        self.sess.run([self.reset_correct, self.val_reset_loss])
        if save:
            if acc > self.best_acc:
                self.logger.info(
                    "Epoch %d has the best accuracy so far: %f, saving..." % (epoch, acc))
                self.best_acc = acc
                self.train_net.save(self.sess, self.savedir, "net_best_acc")
                self.optim_saver.save(self.sess, os.path.join(self.savedir, "optim_best_acc"))
            if loss < self.best_avg_val_loss:
                self.logger.info(
                    "Epoch %d has the best avg_val_loss so far: %f, saving..." % (epoch, loss))
                self.best_avg_val_loss = loss
                self.train_net.save(self.sess, self.savedir, "net_best_loss")
                self.optim_saver.save(self.sess, os.path.join(self.savedir, "optim_best_loss"))

    def summarize(self, epoch):
        avg_loss, cur_summary = self.sess.run([self.train_avg_loss, self.train_summary])
        self.sum_writer.add_summary(cur_summary, epoch)
        self.logger.info(
            "Epoch %d done, avg training loss: %f, evaluating..." % (epoch, avg_loss))
        self.eval(epoch)
        self.sess.run(self.train_reset_loss)
        self.train_net.save(self.sess, self.savedir, "net_latest")
        self.optim_saver.save(self.sess, os.path.join(self.savedir, "optim_latest"))
        with open(os.path.join(self.savedir, "current_status.txt"), "w") as f:
            f.write("%d\n%f\n%f" % (epoch + 1, self.best_acc, self.best_avg_val_loss))
        self.logger.info(
            "Epoch %d done and all ckpts and progress saved to %s" % (epoch, self.savedir))

    def fit(self):
        self.logger.info("Starts training...")
        for epoch in range(self.epoch, self.epochs + 1):
            self.trainset.initialize(self.sess, True)
            self.logger.info("Epoch %d begins..." % epoch)
            for _ in self.tqdm(range(1, len(self.trainset) + 1),
                               desc="[Epoch %d/%d]" % (epoch, self.epochs), ascii=self.ascii):
                self.sess.run([self.train_accum_loss, self.train_op], {
                    self.train_net.is_training: True,
                    self.global_step: epoch})
            self.summarize(epoch)
        self.logger.info("Model fitting done.")

    def export_best(self):
        self.logger.info("Exporting models of best accuracy and loss to optimized frozen pbs...")
        self.export("net_best_acc")
        self.export("net_best_loss")

    def export(self, ckptname):
        shape = (self.input_size, self.input_size)
        with tf.Graph().as_default():
            in_node_name = "img_path"
            img_path = tf.placeholder(tf.string, name=in_node_name)
            # NOTE: decode_jpeg supports png
            x = tf.cast(tf.image.resize_images(tf.expand_dims(
                tf.image.decode_jpeg(tf.read_file(img_path), channels=3), 0), shape), tf.float32)
            x = (x - tf.constant([[[MEAN]]])) / tf.constant([[[STD]]])  # [[[]]] for [n, c, h, w]
            # Hope graph optimization tool may fuse these ops.
            # NOTE: tf.image.resize_images does expand_dims on ndims==3 images and squeeze
            #       back; thus expand_dims first so resize_image would do less things.
            net = Net(x, alpha=self.alpha, input_size=(self.input_size, self.input_size),
                      optim_graph=True)  # optim_graph==True makes inference_only==True
            with tf.Session() as sess:
                net.load(sess, self.savedir, ckptname)
                npypath = os.path.join(self.savedir, "%s.pkl" % ckptname)
                net.save_to_numpy(sess, npypath)
                if self.debug:
                    test_img = os.path.join("datasets", "train", "cat.0.jpg")
                    test_y = sess.run(
                        net.out, {img_path: test_img})
                    out_var_name = net.out.name
                self.logger.info("Params of list of numpy array format saved to %s" % npypath)
                in_graph_def = tf.get_default_graph().as_graph_def()
                out_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, in_graph_def, [net.out.op.name])
            out_graph_def = TransformGraph(out_graph_def, [in_node_name], [net.out.op.name],
                                           ["strip_unused_nodes",
                                            # "fuse_convolutions",
                                            "fold_constants(ignore_errors=true)",
                                            "fold_batch_norms",
                                            "fold_old_batch_norms"])
            ckptpath = os.path.join(self.savedir, "optimized_%s.pb" % ckptname)
            with tf.gfile.GFile(ckptpath, 'wb') as f:
                f.write(out_graph_def.SerializeToString())
            self.logger.info("Optimized frozen pb saved to %s" % ckptpath)
            node_name_path = os.path.join(self.savedir, "node_names.txt")
            if not os.path.exists(os.path.join(node_name_path)):
                with open(node_name_path, "w") as f:
                    f.write("%s\n%s" % (in_node_name, net.out.op.name))
        if self.debug:
            with tf.Graph().as_default():
                gd = tf.GraphDef()
                with tf.gfile.GFile(ckptpath, "rb") as f:
                    gd.ParseFromString(f.read())
                tf.import_graph_def(gd, name="")
                tf.get_default_graph().finalize()
                with tf.Session() as sess:
                    img_path = tf.get_default_graph().get_tensor_by_name("%s:0" % in_node_name)
                    out = tf.get_default_graph().get_tensor_by_name(out_var_name)
                    new_y = sess.run(out, {img_path: test_img})
                diff = np.abs(new_y - test_y)
                self.logger.debug("Diff between original and optimized: %f" % diff)
                self.logger.debug("Diff < 5e-7: %s" % (diff < 5e-7))


def main(**kwargs):
    t = Trainer(**kwargs)
    t.fit()
    t.export_best()


if __name__ == "__main__":
    import argparse
    import sys
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="datasets")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--valset_ratio", type=float, default=.1)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=0.5,
                        choices=[0.5, 1.0])
    parser.add_argument("--optim", type=str, default="adam",
                        choices=list(__optimizers__.keys()))
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--optim_arg1", type=float, default=.9)
    parser.add_argument("--optim_arg2", type=float, default=.999)
    parser.add_argument("--optim_arg3", type=float, default=1e-8,
                        help=(
                            "Note that if you're using sgd optimizer, "
                            "and you're passing this arg greater-equal than 1, "
                            "you're using Nesterov momentum."
                        ))
    parser.add_argument("--lr_decay_step", type=int, default=30)
    parser.add_argument("--lr_decay_rate", type=float, default=.1)
    parser.add_argument(
        "--init_param", type=str, default="imagenet_pretrained_shufflenetv2_0.5.pkl")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--savedir", type=str, default="ckpts")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--logging_lvl", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--logger_out_file", type=str, default=None)
    parser.add_argument("--not_show_progress_bar", action="store_true")
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_tf_cpp_log", action="store_true")
    args = parser.parse_args()

    if not args.show_tf_cpp_log:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args.show_progress = not args.not_show_progress_bar
    log_lvl = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL}
    args.logger = logging.getLogger("Trainer")
    if args.debug:
        args.logger.setLevel(logging.DEBUG)
    else:
        args.logger.setLevel(log_lvl[args.logging_lvl])
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    args.logger.addHandler(stdhandler)
    if args.logger_out_file is not None:
        fhandler = logging.StreamHandler(open(args.logger_out_file, "a"))
        fhandler.setFormatter(formatter)
        args.logger.addHandler(fhandler)
    args.optim_args = [args.optim_arg1, args.optim_arg2, args.optim_arg3]
    del args.optim_arg1, args.optim_arg2, args.optim_arg3, args.show_tf_cpp_log
    del args.not_show_progress_bar, args.logging_lvl, args.logger_out_file
    kwargs = vars(args)
    main(**kwargs)
