import tensorflow as tf 
from tensorflow.python.framework import ops

import numpy as np 

@tf.RegisterGradient("CustomRoundGrad")
def _round_grad(unused_op, grad):
  return grad

def myroundtf(x):
    diff_round = tf.maximum(x-tf.floor(x)-0.49, 0)
    diff_round = diff_round * 100. 
    diff_round = tf.minimum(diff_round, 1)
    return tf.floor(x) + diff_round 

def myquantize(x, gain=None):
    if gain is None:
        return x 
    else:
        x = x * np.float(gain)
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "CustomRoundGrad"}):
            output = tf.identity(myroundtf(x), name="Identity")
            # output = tf.identity(tf.maximum(x,tf.floor(x)), name="Identity")
        return output

def mydequantize(x, gain=None):
    if gain is None:
        return x
    else:
        tx = tf.cast(x / np.float(gain), tf.float32)
        return tx 


# convert float point 32 bit to integer 16 bit 
# need to sure that range of output is not out of bound integer 16 bit
def myconvert_F2I(x, gain=None, method='tf'):
    if gain is None:
        return x 
    else:
        if method == 'tf':
            return tf.floor(x*np.float(gain))
        elif method == 'np':
            return np.floor(x*np.float(gain))

def myconvert_I2F(x, gain=None, method='tf'):
    if gain is None: 
        return x
    else:
        if method == 'tf':
            return tf.cast(x/np.float(gain), tf.float32)
        elif method == 'np':
            return x/np.float(gain)


def tf_floor(x, name=None):
    """Differentiable floor based in numpy
    Args
        x: first argument
    Returns
        floor of x
    """

    def np_floor(x):
        return np.floor(x, dtype=np.float32)

    def floorgrad(op, grad):
        x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )

        return grad * 1#the propagated gradient with respect to the first and second argument respectively

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    with ops.name_scope(name, "floor", [x]) as name:
        z = py_func(np_floor,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=floorgrad)  # <-- here's the call to the gradient
        return tf.reshape(z[0], tf.shape(x)) 

def tf_round(x, name=None):
    """Differentiable floor based in numpy
    Args
        x: first argument
    Returns
        floor of x
    """

    def np_round(x):
        return np.round(x)

    def roundgrad(op, grad):
        x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )

        return grad * 1#the propagated gradient with respect to the first and second argument respectively

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    with ops.name_scope(name, "round", [x]) as name:
        z = py_func(np_round,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=roundgrad)  # <-- here's the call to the gradient
        return tf.reshape(z[0], tf.shape(x)) 

def tf_quantize(x, gain=1, name=None):
    return tf_floor(x*gain, name=name)

def tf_dequantize(x, gain=1):
    print(tf.shape(x))
    x = x / gain 
    return x 

# get gain scale to gain x [-a,b] to [-maxrange, maxrange-1]
def tf_get_gain_to_range(x, maxrange=127):
    maxabs = tf.reduce_max(tf.abs(x))
    return maxrange / maxabs
