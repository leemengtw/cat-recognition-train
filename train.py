import os
from datetime import datetime
import tensorflow as tf
from dataset import Dataset
from net import Net


class Trainer():

    def __init__(
            self,
            data_folder="dataset",
            batch_size=64,
            input_size=224,
            valset_ratio=.1,
            epochs=400,
            init_lr=1e-3,
            init_params=None,
            logdir="runs",
            savedir="ckpts",
            random_seed=0,
            logger=None):
        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.info)
            self.logger.warning("You are using the root logger, which has bad a format.")
            self.logger.warning("Please consider passing a better logger.")
        subdir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.savedir = os.path.join(savedir, subdir)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
        with open(os.path.join(self.savedir, "history"), "a") as fp:
            fp.write("%s\n" % subdir)
        self.logdir = os.path.join(logdir, subdir)
        self.best_avg_val_loss, self.best_acc = float("inf"), 0.
        self.logger.info("Preparing dataset...")
        self.trainset = Dataset(
            data_folder, True, (input_size, input_size),
            batch_size, None, epochs, valset_ratio, random_seed)
        self.total_pred = tf.constant(self.trainset.val_length, name="total_pred")
        self.logger.info("Generating training operations...")
        train_x, train_y = self.trainset.train_iterator.get_next()
        self._build_train_graph(train_x, train_y, init_lr, init_params)
        self.logger.info("Generating validation operations...")
        val_x, val_y = self.trainset.val_iterator.get_next()
        self._build_val_graph(val_x, val_y)
        self.sess = tf.Session()
        self.sum_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_train_graph(self, train_x, train_y, init_lr, init_params=None):
        self.train_net = Net(train_x, init_params=init_params)
        loss, self.train_accum_loss, self.train_avg_loss, self.train_reset_loss = \
            self._build_loss_graph(self.train_net.out, train_y, "train")
        self.train_op = tf.train.AdamOptimizer(
                learning_rate=init_lr).minimize(loss)
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
        self.trainset.initialize(self.sess)  # inits valset inside trainset
        for i in range(self.trainset.val_total_batches):
            self.sess.run(
                [self.accum_correct, self.val_accum_loss], {self.val_net.is_training: False})
        acc, loss, summary = self.sess.run([self.accuracy, self.val_avg_loss, self.val_summary])
        self.logger.info(
            "Epoch %d evaluation done, acc: %f, avg_val_loss: %f" % (epoch, acc, loss))
        self.sum_writer.add_summary(summary, epoch)
        self.sess.run([self.reset_correct, self.val_reset_loss])
        if save:
            if acc > self.best_acc:
                self.logger.info("Epoch %d has currently best accuracy: %f" % (epoch, acc))
                self.best_acc = acc
                self.train_net.save(self.sess, self.savedir, "best_acc")
            if loss < self.best_avg_val_loss:
                self.logger.info("Epoch %d has currently best avg_val_loss: %f" % (epoch, loss))
                self.best_avg_val_loss = loss
                self.train_net.save(self.sess, self.savedir, "best_avg_val_loss")

    def summarize(self, epoch):
        avg_loss, cur_summary = self.sess.run([self.train_avg_loss, self.train_summary])
        self.sum_writer.add_summary(cur_summary, epoch)
        self.logger.info(
            "Epoch %d done, avg training loss: %f, evaluating..." % (epoch, avg_loss))
        self.eval(epoch)
        self.sess.run(self.train_reset_loss)
        self.train_net.save(self.sess, self.savedir, "latest")

    def fit(self):
        train_batch_per_epoch = self.trainset.train_batch_per_epoch
        self.logger.info("Starts training...")
        for i in range(self.trainset.train_total_batches):
            self.sess.run([
                self.train_accum_loss, self.train_op], {self.train_net.is_training: True})
            if i % train_batch_per_epoch + 1 == train_batch_per_epoch:
                epoch = i // train_batch_per_epoch + 1
                self.summarize(epoch)


def main(*args):
    t = Trainer(*args)
    t.fit()


if __name__ == "__main__":
    import argparse
    import sys
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="datasets")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--valset_ratio", type=float, default=.1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument(
        "--init_params", type=str, default="imagenet_pretrained_shufflenetv2_1.0.pkl")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--savedir", type=str, default="ckpts")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--logging_lvl", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--logger_out_file", type=str, default=None)
    parser.add_argument("--show_tf_cpp_log", action="store_true")
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

    main(args.data_folder, args.batch_size, args.input_size, args.valset_ratio, args.epochs,
         args.init_lr, args.init_params, args.logdir, args.savedir, args.random_seed, logger)
