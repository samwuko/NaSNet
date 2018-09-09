import tensorflow as tf
import misc, config, nasnet

data_dir  = "../splits/"
data_list = data_dir + "train-19zl.csv"

def main(_):

  
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  # config = tf.ConfigProto(gpu_options=gpu_options)
  # with tf.Session(config=config) as sess:

    network = nasnet.CrossNet(sess)
    network.load_data(data_list, data_dir)

    misc.pprint(network.config.__flags)

    network.train_test()

if __name__ == '__main__':
  tf.app.run()
