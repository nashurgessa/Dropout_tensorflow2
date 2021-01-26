import numbers
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    pass


def get_noise_shape(x, noise_shape):
    if noise_shape is None:
        return tf.shape(x)
    
    try:
        noise_shape_ = tf.shape(noise_shape)
    except(TypeError, ValueError):
        return noise_shape
    
    if x.shape is not None and len(x.shape) == len(noise_shape_):
        new_dims = []
        for i, dim in enumerate(x.shape):
            if noise_shape_[i].value is None and dim.value is not None:
                new_dims.append(dim.value)
            else:
                new_dims.append(noise_shape_[i].value)
        return tf.shape(new_dims)
         
    return noise_shape


def nn_dropout(x, rate, noise_shape=None, seed=None, name=None):
    """
    Dropout: Randomly sets elements to zero to prevent overfitting
    
    [Dropout](https://arxiv.org/abs/1207.0580) is usedful for 
    regularizing the DNN models. Inputs elements are randomly set to
    zero (and the other elements are rescaled)
    
    More precisely: with probability 'rate' elements of 'x' are
    set to '0'. The remaining elements are scaled up by the
     1.0 / (1.0 - rate), so that the expected value is preserved 
    
    """
    
    with tf.name_scope("dropout") as name:
        is_rate_number = isinstance(rate, numbers.Real)
        if is_rate_number and (rate <0 or rate >=1):
            raise ValueError("rate must be a scalar tensor or a flot " 
                             "in the range [0, 1), got %g" %rate)
        
        x = tf.convert_to_tensor(x, name="x")
        
        x_dtype = x.dtype
        if not x_dtype.is_floating:
            raise ValueError("x has to be floating point tensor "
                             "it is going to be scaled. Got a %s "
                             " tensor instead." %x_dtype)
            
        is_executing_eagerly = tf.executing_eagerly()  # Check if execute eagearly
        
        if not tf.is_tensor(rate):
            if is_rate_number:
                keep_prob = 1 - rate
                scale = 1 / keep_prob
                scale = tf.convert_to_tensor(scale, dtype=x_dtype)
                ret = tf.multiply(x, scale)
            else:
                raise ValueError("rate is neither scalar nor "
                                 "scalar tensor %r" %rate)
        
        else:
            raise ValueError("Tensor dtype %s is incomptaible "
                             "with Tensor dtype %s: %r " %(
                                 x_dtype.name, rate.dtype.name, rate
                             ))
        
        noise_shape = get_noise_shape(x, noise_shape)

        random_tensor = tf.random.uniform(noise_shape, seed=seed, dtype=x_dtype)
        
        keep_mask = random_tensor >= rate
        ret = tf.multiply(ret, tf.cast(keep_mask, dtype=x_dtype))
        
        if not is_executing_eagerly:
            ret.set_shape(x.get_shape())
        
        return ret

class Drop_out_layer(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Drop_out_layer, self).__init__()
        self.rate = rate
        # if isinstance(self.rate, (int, float)) and not rate:
        #     keras_temporary_dropout_rate.get_cell().set(True)
        # else:
        #     keras_temporary_dropout_rate.get_cell().set(False)
        self.noise_shape = noise_shape
        self.seed = seed
        self.supporting_masking = True
        
    def get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None
        
        concrete_input_shape = tf.shape(inputs)
        noise_shape= []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_input_shape[i] if value is None else value)    
        return tf.convert_to_tensor(noise_shape)
      
        
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()    
        
        def dropped_inputs():
            return nn_dropout(inputs, 
                              noise_shape=self.get_noise_shape(inputs),
                              seed=self.seed,
                              rate=self.rate)
        
        output = tf.cond(training, dropped_inputs, lambda: tf.identity(inputs))
        
        return output
        
def main():
    Drop_out_layer(rate=0.5)


if __name__ == '__main__':
    main()
    tf.random.set_seed(0)
    x = tf.ones([3, 5])
    print("Input_data")
    print(x.numpy())
    y = nn_dropout(x, rate=0.5)
    print("Dropout output")
    print(y.numpy())
    x = tf.ones([3, 10])
    print("Input_data")
    print(x.numpy())
    y = nn_dropout(x, rate=2/3, noise_shape=[1,10], seed=1)
    print("Dropout output")
    print(y.numpy())
    print("------------------------------------------------ nn.Layer ------------------------------------------------")
    layer = Drop_out_layer(.2, input_shape=(2,))
    data = np.arange(10).reshape(5, 2).astype(np.float32)
    print("Input_data: ")
    print(data)
    outputs = layer(data, training=True)
    print("Dropout Layer")
    print(outputs.numpy())