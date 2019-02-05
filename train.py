import os
import argparse
from datetime import datetime
import tensorflow as tf
from dataset import Dataset
from net import Net

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--logdir", type=str, default="runs")
parser.add_argument("--savedir", type=str, default="ckpts")
parser.add_argument("--net_input_size", type=int, default=224)
parser.add_argument("--data_folder", type=str, default="datasets")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--initial_learning_rate", type=float, default=1e-3)
parser.add_argument("--valset_ratio", type=float, default=.1)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument(
    "--initial_weights", type=str, default="imagenet_pretrained_shufflenetv2_1.0.pkl")
args = parser.parse_args()


class Trainer():

    def __init__(self):
        subdir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.savedir = os.path.join(args.savedir, subdir)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
        with open(os.path.join(self.savedir, "history"), "a") as fp:
            fp.write("%s\n" % subdir)
        self.logdir = os.path.join(args.logdir, subdir)
        self.best_loss, self.best_acc = float("inf"), 0.
        self.trainset = Dataset(
            args.data_folder, True, (args.net_input_size, args.net_input_size),
            args.batch_size, None, args.epochs, False, args.valset_ratio, args.random_seed)
        self.total_pred = tf.constant(self.trainset.val_length, name="total_pred")
        train_x, train_y = self.trainset.train_iterator.get_next()
        self._build_train_graph(train_x, train_y)
        val_x, val_y = self.trainset.val_iterator.get_next()
        self._build_val_graph(val_x, val_y)
        self.sess = tf.Session()
        self.sum_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_train_graph(self, train_x, train_y):
        self.train_net = Net(train_x, init_params=args.initial_weights)
        loss, self.train_accum_loss, train_avg_loss, self.train_reset_loss = \
            self._build_loss_graph(self.train_net.out, train_y, "train")
        self.train_op = tf.train.AdamOptimizer(
                learning_rate=args.initial_learning_rate).minimize(loss)
        self.train_summary = tf.summary.scalar("training loss", train_avg_loss)

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
        self.sum_writer.add_summary(summary, epoch)
        self.sess.run([self.reset_correct, self.val_reset_loss])
        if save:
            if acc > self.best_acc:
                self.best_acc = acc
                self.train_net.save(self.sess, self.savedir, "best_acc")
            if loss < self.best_loss:
                self.best_loss = loss
                self.train_net.save(self.sess, self.savedir, "best_loss")

    def fit(self):
        train_batch_per_epoch = self.trainset.train_batch_per_epoch
        for i in range(self.trainset.train_total_batches):
            self.sess.run([
                self.train_accum_loss, self.train_op], {self.train_net.is_training: True})
            if i % train_batch_per_epoch + 1 == train_batch_per_epoch:
                epoch = i // train_batch_per_epoch + 1
                self.sum_writer.add_summary(self.sess.run(self.train_summary), epoch)
                self.eval(epoch)
                self.sess.run(self.train_reset_loss)
                self.train_net.save(self.sess, self.savedir, "latest")


def main():
    t = Trainer()
    t.fit()


if __name__ == "__main__":
    main()
