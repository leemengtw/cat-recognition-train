import os
import pickle
from datetime import datetime
import tensorflow as tf
from dataset import Dataset
from net import Net

__no_tqdm__ = False
try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    __no_tqdm__ = True


def _tqdm(res):
    return res


class Trainer():

    def __init__(
            self,
            data_folder="dataset",
            batch_size=8,
            input_size=224,
            valset_ratio=.1,
            epochs=90,
            init_lr=1e-3,
            init_params=None,
            logdir="runs",
            savedir="ckpts",
            random_seed=0,
            logger=None,
            show_progress=True,
            restore=None):
        self.exported = False
        self.best_avg_val_loss, self.best_acc = float("inf"), 0.
        self.epoch = 1
        subdir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.logdir = os.path.join(logdir, subdir)
        self.savedir = os.path.join(savedir, subdir)
        if restore is not None:
            with open(os.path.join(restore, "config.pkl"), "rb") as f:
                (data_folder, batch_size, input_size, valset_ratio,
                 epochs, self.best_avg_val_loss, self.best_acc,
                 self.logdir, self.savedir) = pickle.load(f)
            with open(os.path.join(restore, "current_epoch.txt"), "r") as f:
                self.epoch = int(f.read())
            if os.path.exists(os.path.join(restore, "net_latest.meta")) and\
                    (init_params is not None):
                self.logger.warning(
                    "Checkpoint in %s exists; not initializing params from %s" % (
                        restore, init_params))
                init_params = None
        else:
            if not os.path.isdir(self.savedir):
                os.makedirs(self.savedir)
            with open(os.path.join(savedir, "history"), "a") as fp:
                fp.write("%s\n" % subdir)
            with open(os.path.join(self.savedir, "config.pkl"), "wb") as f:
                pickle.dump(
                    (data_folder, batch_size, input_size, valset_ratio,
                     epochs, self.best_avg_val_loss, self.best_acc,
                     self.logdir, self.savedir),
                    f)
            with open(os.path.join(self.savedir, "current_epoch.txt"), "w") as f:
                f.write(str(self.epoch))
        if not show_progress or __no_tqdm__:
            self.tqdm = _tqdm
        else:
            self.tqdm = tqdm
        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.info)
            self.logger.warning("You are using the root logger, which has bad a format.")
            self.logger.warning("Please consider passing a better logger.")
        tf.set_random_seed(random_seed)
        self.epochs = epochs
        self.logger.info("Model checkpoints will be saved to %s" % self.savedir)
        self.logger.info("Summary for TensorBaord will be saved to %s" % self.logdir)
        self.logger.info(
            "You can use \"tensorboard --logdir %s\" to see all training summaries." % logdir)
        # logdir: folder containing all training histories
        # self.logdir: folder containing current training summary
        self.logger.info("Preparing dataset...")
        self.trainset = Dataset(
            data_folder, True, (input_size, input_size),
            batch_size, None, valset_ratio, random_seed)
        self.input_size = input_size
        self.logger.debug("%d training instances, %d validation instances" % (
            self.trainset.train_length, self.trainset.val_length))
        self.total_pred = tf.constant(self.trainset.val_length, name="total_pred")
        self.logger.info("Generating training operations...")
        x, y = self.trainset.get_next()
        self._build_train_graph(x, y, init_lr, init_params)
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

    def _build_train_graph(self, train_x, train_y, init_lr, init_params=None):
        self.train_net = Net(train_x, init_params=init_params)
        loss, self.train_accum_loss, self.train_avg_loss, self.train_reset_loss = \
            self._build_loss_graph(self.train_net.out, train_y, "train")
        with tf.variable_scope("optim_vars"):
            self.train_op = tf.train.AdamOptimizer(
                    learning_rate=init_lr).minimize(loss)
        self.optim_vars = [
            v for v in tf.global_variables() if v.name.startswith("optim_vars")]
        self.optim_saver = tf.train.Saver(self.optim_vars)
        self.train_summary = tf.summary.scalar("training loss", self.train_avg_loss)

    def _build_val_graph(self, val_x, val_y):
        self.val_net = Net(val_x, reuse=True)  # no need to pass init_params since it's reused
        cur_currect = tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(self.val_net.out > 0, tf.int32), val_y), tf.int32))
        total_correct = tf.get_variable("total_correct", None, tf.int32, tf.constant(0))
        self.accum_correct = tf.assign(total_correct, total_correct + cur_currect)
        self.reset_correct = tf.assign(total_correct, 0)
        self.accuracy = total_correct / self.total_pred
        _, self.val_accum_loss, self.val_avg_loss, self.val_reset_loss = \
            self._build_loss_graph(self.val_net.out, val_y, "val")
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
        if self.exported:
            raise Exception("%s %s" % (
                    "Model exported and all graphs are reset; ",
                    "please reinitialize the trainer if you want to evaluate the model."))
        self.trainset.initialize(self.sess, False)  # inits valset inside trainset
        for _ in self.tqdm(range(len(self.trainset)), desc="[Epoch %d Evaluation]" % epoch):
            self.sess.run(
                [self.accum_correct, self.val_accum_loss], {self.val_net.is_training: False})
        acc, loss, summary = self.sess.run([self.accuracy, self.val_avg_loss, self.val_summary])
        self.logger.info(
            "Epoch %d evaluation done, acc: %f, avg_val_loss: %f" % (epoch, acc, loss))
        self.sum_writer.add_summary(summary, epoch)
        self.sess.run([self.reset_correct, self.val_reset_loss])
        if save:
            if acc > self.best_acc:
                self.logger.info("Epoch %d has the best accuracy so far: %f" % (epoch, acc))
                self.best_acc = acc
                self.train_net.save(self.sess, self.savedir, "net_best_acc")
                self.optim_saver.save(self.sess, os.path.join(self.savedir, "optim_best_acc"))
            if loss < self.best_avg_val_loss:
                self.logger.info("Epoch %d has the best avg_val_loss so far: %f" % (epoch, loss))
                self.best_avg_val_loss = loss
                self.train_net.save(self.sess, self.savedir, "net_best_loss")
                self.optim_saver.save(self.sess, os.path.join(self.savedir, "optim_best_loss"))

    def summarize(self, epoch):
        if self.exported:
            raise Exception("%s %s" % (
                    "Model exported and all graphs are reset; ",
                    "please reinitialize the trainer if you want to summarize the model."))
        avg_loss, cur_summary = self.sess.run([self.train_avg_loss, self.train_summary])
        self.sum_writer.add_summary(cur_summary, epoch)
        self.logger.info(
            "Epoch %d done, avg training loss: %f, evaluating..." % (epoch, avg_loss))
        self.eval(epoch)
        self.sess.run(self.train_reset_loss)
        self.train_net.save(self.sess, self.savedir, "net_latest")
        self.optim_saver.save(self.sess, os.path.join(self.savedir, "optim_latest"))
        with open(os.path.join(self.savedir, "current_epoch.txt"), "w") as f:
            f.write(str(epoch + 1))
        self.logger.info(
            "Epoch %d done and all ckpts and progress saved to %s" % (epoch, self.savedir))

    def fit(self):
        if self.exported:
            raise Exception("%s %s" % (
                    "Model exported and all graphs are reset; ",
                    "please reinitialize the trainer if you want to continue training."))
        self.logger.info("Starts training...")
        for epoch in range(self.epoch, self.epochs + 1):
            self.trainset.initialize(self.sess, True)
            self.logger.info("Epoch %d begins..." % epoch)
            for _ in self.tqdm(range(1, len(self.trainset) + 1),
                               desc="[Epoch %d/%d]" % (epoch, self.epochs)):
                self.sess.run([
                    self.train_accum_loss, self.train_op], {self.train_net.is_training: True})
            self.summarize(epoch)
        self.logger.info("Model fitting done.")

    def export_best(self):
        self.export("net_best_acc")
        self.export("net_best_loss")

    # TODO: finish this!
    def export(self, fname):
        self.exported = True
        tf.reset_default_graph()
        # x = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3], name="x")
        # net = Net(x, inference_only=True)
        # FIXME


def main(*args):
    t = Trainer(*args)
    t.fit()


if __name__ == "__main__":
    import argparse
    import sys
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="datasets")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--valset_ratio", type=float, default=.1)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument(
        "--init_params", type=str, default="imagenet_pretrained_shufflenetv2_1.0.pkl")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--savedir", type=str, default="ckpts")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--logging_lvl", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--logger_out_file", type=str, default=None)
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--show_tf_cpp_log", action="store_true")
    parser.add_argument("--not_show_progress_bar", action="store_true")
    args = parser.parse_args()

    if not args.show_tf_cpp_log:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    log_lvl = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL}
    logger = logging.getLogger("Trainer")
    logger.setLevel(log_lvl[args.logging_lvl])
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.logger_out_file is not None:
        fhandler = logging.StreamHandler(open(args.logger_out_file, "a"))
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    main(
        args.data_folder, args.batch_size, args.input_size, args.valset_ratio,
        args.epochs, args.init_lr, args.init_params, args.logdir,
        args.savedir, args.random_seed, logger, not args.not_show_progress_bar, args.restore)
