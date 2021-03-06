
import tensorflow as tf
 
 
def transformer(src, theta, out_size, name='SpatialTransformer', **kwargs):
    #print('beigin-transformer')
  
    
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            y = tf.reshape(x,[-1,1])
            z = tf.cast(tf.ones([1,n_repeats]),tf.int32)
            #rep = tf.transpose(
            #    tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            #rep = tf.cast(rep, 'int32')
            #x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(y*z, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            #x = tf.cast(x, tf.float32)
            #y = tf.cast(y, 'float32')
            height_f = tf.cast(height, tf.float32)
            width_f = tf.cast(width, tf.float32)
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype=tf.int32)

            max_y = tf.cast(tf.shape(im)[1] - 1, tf.int32)
            max_x = tf.cast(tf.shape(im)[2] - 1, tf.int32)

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x = tf.reshape(x,[-1])
            y = tf.reshape(y,[-1])

            x0 = tf.cast(tf.floor(x), tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            line = width
            size = height*width
            out_size = out_height*out_width
            base = _repeat(tf.range(num_batch)*size, out_size)
            #base = tf.reshape(tf.transpose(tf.reshape(tf.tile(tf.range(num_batch)*size,[out_size]),[out_size,num_batch])),[-1])
            base_y0 = base + y0*line
            base_y1 = base + y1*line
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, [-1,channels])
            im_flat = tf.cast(im_flat, tf.float32)
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, tf.float32)
            x1_f = tf.cast(x1, tf.float32)
            y0_f = tf.cast(y0, tf.float32)
            y1_f = tf.cast(y1, tf.float32)

            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        #print('begin--meshgrid')
        with tf.variable_scope('_meshgrid'):
            x = tf.linspace(-1.0,1.0,width)  #cast(tf.range(0,width),tf.float32)
            y = tf.linspace(-1.0,1.0,height) #cast(tf.range(0,height),tf.float32)

            x_t,y_t = tf.meshgrid(x,y)
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            """
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            print('meshgrid_x_t_ok')
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))
            print('meshgrid_y_t_ok')
            """
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            #print('meshgrid_flat_t_ok')
            ones = tf.ones_like(x_t_flat)
            #change the the final activation function
            #h_w_ = ones*height #!!!!!!!!!!!!!!!!!!!!!
            #print('meshgrid_ones_ok')
            #print(x_t_flat)
            #print(y_t_flat)
            #print(ones)

            grid = tf.concat([x_t_flat, y_t_flat, ones],0)
            #print ('over_meshgrid')
            return grid

    def _transform(theta, images, out_size):
        #print('_transform')

        with tf.variable_scope('_transform'):
            num_batch = tf.shape(images)[0]
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]
            num_channels = tf.shape(images)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            #grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, [num_batch])
            grid = tf.reshape(grid, [num_batch, 3, -1])
            #tf.batch_matrix_diag
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            #print('begin--batch--matmul')
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            #x_s_flat = tf.reshape(x_s, [-1])
            #y_s_flat = tf.reshape(y_s, [-1])
            input_transformed = _interpolate(images, x_s, y_s,out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            #print('over_transformer')
            return output

    with tf.variable_scope(name):
        output = _transform(theta, src, out_size)
        return output

def batch_transformer(images, thetas, out_size, name='BatchSpatialTransformer'):

    with tf.variable_scope(name):
        #num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        #indices = [[i]*num_transforms for i in xrange(num_batch)]
        #input_repeated = tf.gather(images, tf.reshape(indices, [-1]))
        return transformer(images, thetas, out_size)
