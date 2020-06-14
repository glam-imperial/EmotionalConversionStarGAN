"""
logger.py

Altered version of Logger.py by hujinsen. Original source can be found at:
    https://github.com/hujinsen/pytorch-StarGAN-VC
"""
import tensorflow as tf
import os


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir, model_name):
        """Initialize summary writer."""

        file_dir = os.path.join(log_dir, model_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        self.writer = tf.summary.FileWriter(file_dir)
        print("Made logger.")

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
