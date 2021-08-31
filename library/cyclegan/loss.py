import tensorflow as tf


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

default_disc_alpha = 5
default_gen_alpha = 5
default_cycle_alpha = 7
default_identity_beta = 1.5


def discriminator_loss(real_images, generated_images, alpha):
    real_loss = bce(tf.ones_like(real_images), real_images)
    gen_loss = bce(tf.zeros_like(generated_images), generated_images)
    return (real_loss + gen_loss) * alpha


def generator_loss(generated_images, alpha):
    return bce(tf.ones_like(generated_images), generated_images) * alpha


def cycle_loss(real_images, cycled_images, alpha):
    return tf.reduce_mean(tf.abs(real_images - cycled_images)) * alpha


def identity_loss(real_images, generated_image_of_same_type, alpha, beta):
    return tf.reduce_mean(tf.abs(real_images - generated_image_of_same_type)) * alpha * beta
