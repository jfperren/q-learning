import io
import numpy as np
from PIL import Image
import tensorflow as tf

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir)

    def create_folder(self):
        os.makedirs(self.logdir, exist_ok=True)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        
    def log_histogram(self, tag, values, step, bins):
        with self.writer.as_default():
            values = np.array(values)
            counts, bin_edges = np.histogram(values, bins=bins)
            tf.summary.histogram(tag, counts, step=step)
            self.writer.flush()

    def log_image(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
