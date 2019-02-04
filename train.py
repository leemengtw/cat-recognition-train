import argparse
import tensorflow as tf
from dataset import Dataset
from net import Net

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--net_input_size", type=int, default=224)
parser.add_argument("--data_folder", type=str, default="datasets")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--initial_learning_rate", type=float, default=1e-3)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--valset_ratio", type=float, default=.1)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument(
    "--initial_weights", type=str, default="imagenet_pretrained_shufflenetv2_1.0.pkl")
args = parser.parse_args()


class Trainer():

    def __init__(self):
        self.trainset = Dataset(
            args.data_folder, True, (args.net_input_size, args.net_input_size),
            args.batch_size, None, args.epochs, False, args.valset_ratio, args.random_seed)
        train_x, train_y = self.trainset.train_iterator.get_next()
        val_x, val_y = self.trainset.val_iterator.get_next()
        self.train_net = Net(train_x, init_params=args.initial_weights)
        self.val_net = Net(val_x, reuse=True)  # no need to pass init_params since it's reused
        train_logits = self.train_net.out
        val_logits = self.val_net.out
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=train_y, logits=train_logits))
        self.train_op = tf.train.AdamOptimizer(
                learning_rate=args.initial_learning_rate).minimize(self.loss)
        self.correct_preds = tf.equal(tf.argmax(val_logits, 1), tf.argmax(val_y, 1))
        # TODO: finish validation; how to accumulate correct_preds for accumulative accuracy
        #       and be thrown to tensorboard summary?
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.train_net.is_training: True})
        # FIXME
        print(loss)

    def eval(self):
        pass

    def fit(self):
        self.train()


def main():
    t = Trainer()
    t.fit()


if __name__ == "__main__":
    main()
