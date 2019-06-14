# tf_data_benchmark
+ With `tf.data`:

Augmentations=[GoogleNetResize, Flip]: 11k image/s on a DGX1.
As a reference, a DGX1 with 8 V100s can train ResNet-50 in fp32 at 2.6k image/s.
```
python benchmark-tfdata.py /path/to/imagenet --name train --batch 128
```
