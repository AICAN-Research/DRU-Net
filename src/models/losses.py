import tensorflow as tf


def get_dice_loss(nb_classes=1, use_background=False):
    def dice_loss(target, output, epsilon=1e-10):
        smooth = 1.
        dice = 0
        for i in range(0 if use_background else 1, nb_classes):
            output1 = output[..., i]
            target1 = target[..., i]
            intersection1 = tf.reduce_sum(output1 * target1)
            union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
            dice += (2. * intersection1 + smooth) / (union1 + smooth)
        if use_background:
            dice /= nb_classes
        else:
            dice /= (nb_classes - 1)
        return tf.clip_by_value(1. - dice, 0., 1. - epsilon)
    return dice_loss


def dsc_thresholded(nb_classes=2, use_background=False):
    def dice(target, output, epsilon=1e-10):
        smooth = 1.
        dice = 0
        output = tf.cast(output > 0.5, tf.float32)
        for i in range(0 if use_background else 1, nb_classes):
            output1 = output[:,:,:, i]
            target1 = target[:,:,:, i]
            intersection1 = tf.reduce_sum(output1 * target1)
            union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
            dice += (2. * intersection1 + smooth) / (union1 + smooth)
        if use_background:
            dice /= nb_classes
        else:
            dice /= (nb_classes - 1)
        return tf.clip_by_value(dice, 0., 1. - epsilon)
    return dice
