
def binception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 #bilinear=False,
                 scope='InceptionV1'):
  """Defines the Inception V1 architecture.

  This architecture is defined in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  if is_training == False:
    dropout_keep_prob = 1.0
  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV1', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        nc = 1024#256
        #nc = 256
        #net = slim.conv2d(net, nc, [1, 1], scope='Conv2d_zh_1x1')
        #net = slim.dropout(net,0.5, scope='zh_Dropout_1b')
        #net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        #net = slim.batch_norm(net, scope='postnorm')

        print('Shape of conv output', net.get_shape())
        net = tf.einsum('ijkm,ijkn->imn',net,net)
        print('Shape after bilinear', net.get_shape())

        net = tf.reshape(net,[-1,nc*nc])
        #net = tf.divide(net,7.0*7.0)
        net = tf.divide(net,14.0*14.0)
        net = tf.multiply(tf.sign(net),tf.sqrt(tf.abs(net)+1e-12))
        print('Shape before normalization', net.get_shape())
        net = tf.nn.l2_normalize(net, dim=1)
        net = tf.reshape(net,[-1,1,1,nc*nc])
        print('Shape after bilinear norm', net.get_shape())
        net = slim.conv2d(net, 256, [1, 1], scope='fc_1x1')
        net = slim.dropout(net,0.4, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_0c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

binception_v1.default_image_size = 448
