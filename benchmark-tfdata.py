#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-tfdata.py

import tqdm
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile

from symbolic_imagenet import get_imglist, build_pipeline


def get_default_sess_config(mem_fraction=0.99):
    """
    Return a tf.ConfigProto to use as default session config.
    You can modify the returned config to fit your needs.

    Args:
        mem_fraction(float): see the `per_process_gpu_memory_fraction` option
            in TensorFlow's GPUOptions protobuf:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto

    Returns:
        tf.ConfigProto: the config to use.
    """
    conf = tf.ConfigProto()

    conf.allow_soft_placement = True
    # conf.log_device_placement = True

    conf.intra_op_parallelism_threads = 1
    conf.inter_op_parallelism_threads = 0
    # TF benchmark use cpu_count() - gpu_thread_count(), e.g. 80 - 8 * 2
    # Didn't see much difference.

    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction

    # This hurt performance of large data pipeline:
    # https://github.com/tensorflow/benchmarks/commit/1528c46499cdcff669b5d7c006b7b971884ad0e6
    # conf.gpu_options.force_gpu_compatible = True

    conf.gpu_options.allow_growth = True

    # from tensorflow.core.protobuf import rewriter_config_pb2 as rwc
    # conf.graph_options.rewrite_options.memory_optimization = \
    #     rwc.RewriterConfig.HEURISTICS

    # May hurt performance?
    # conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # conf.graph_options.place_pruned_graph = True
    return conf


def benchmark_ds(ds, count, warmup=200):
    itr = ds.make_initializable_iterator()
    dpop = itr.get_next()
    #dp = itr.get_next()
    #dpop = tf.group(*dp)
    with tf.Session(config=get_default_sess_config()) as sess:

        sess.run(itr.initializer)
        for _ in tqdm.trange(warmup):
            sess.run(dpop)
        for _ in tqdm.trange(count, smoothing=0.1):
            sess.run(dpop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='directory to imagenet')
    parser.add_argument('--name', choices=['train', 'val'], default='train')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--parallel', type=int, default=40)
    args = parser.parse_args()

    #imglist = get_imglist(args.data, args.name)
    #print("Number of Images: {}".format(len(imglist)))
    #imglist = gfile.Glob("{}/{}*".format(args.data, args.name))

    with tf.device('/cpu:0'):
        data = build_pipeline(
            args.data, args.name == 'train',
            args.batch, args.parallel)
        if args.name != 'train':
            data = data.repeat()    # for benchmark
    benchmark_ds(data, 1000)
