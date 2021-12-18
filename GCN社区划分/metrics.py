import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    pre = tf.argmax(preds, 1)  #test的估计，但1036维
    lab = tf.argmax(labels, 1) #test的label
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    mask /= tf.reduce_mean(mask) #1/（100/1036）
    accuracy_all *= mask
    #return tf.reduce_mean(accuracy_all),pre,lab,accuracy_all
    return tf.reduce_mean(accuracy_all),pre,lab
