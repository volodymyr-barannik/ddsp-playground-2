import sparsenet
import tensorflow as tf
from sparsenet.core import sparse

from ddsp.training import nn
from tensorflow import float32


def f2():
    inputs = [tf.random.uniform(shape=(1, 256, 1)), tf.random.uniform(shape=(1, 256, 1))]
    x = tf.random.uniform(shape=(1, 256, 256))

    print(inputs)
    print(x)
    x = tf.concat(inputs + [x], axis=-1)

    print(x)


def f1():
    x = tf.random.uniform(shape=(1, 10, 1))
    print(f"x={x}, shape={x.shape}")
    m = tf.keras.Sequential(layers=[tf.keras.layers.Dense(units=4)])

    print(f"\nm has {len(m.layers)} layers: ")
    for layer in m.layers:
        print(f"--\t{layer.name}")

    m(x)
    print(f"\nm={m.layers[0].weights}, shape={m.layers[0].weights}")

    x = m(x)
    print(f"\nm(x)={x}, shape={x.shape}")

    #sprs = nn.sparse(units=8, density=0.2, dtype=float32)
    #msprs = tf.keras.Sequential(layers=[nn.sparse(units=8, density=0.2)])

    print(tf.__version__)
    t = tf.random.uniform(shape=(1, 10))
    x = sparse(units=8, density=0.1, activation=None)(t)
    print(f"sprs(x)={x}, shape={x.shape}")

    #fc = nn.Fc(ch=8)
    #x=fc(x)
    #print(f"fc(x)={x}, shape={x.shape}")

    #dns = tf.keras.layers.Dense(units=4)
    #x = dns(x)
    #print(f"\ndns(x)={x}, shape={x.shape}")

    #fcs = nn.FcStack(8, layers=3, density=1)
    #x=fcs(x)
    #print(f"fcs(x)={x}, shape={x.shape}")


f1()
