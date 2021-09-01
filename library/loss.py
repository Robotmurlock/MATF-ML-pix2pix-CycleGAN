import tensorflow as tf
from tensorflow import keras


bce = keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target, alpha=1, loss_func=None):
    if loss_func is None:
        loss_func = bce

    # Kaznjava se Generator za svaku pogodjenu sliku Diskriminatora
    # Idealno za Generator je da Diskriminator kaže da su sve prave slike tj. jedinice.
    gan_loss = loss_func(tf.ones_like(disc_generated_output), disc_generated_output)
    # Dodaje se L1 greska kako bi se forsiralo da slike izgledaju vise kao obicne
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    # 'alpha' parametar nam omogucava akcentovanja na neku od gresaka
    total_gen_loss = gan_loss + (alpha * l1_loss)
    return total_gen_loss, gan_loss, alpha * l1_loss


def discriminator_loss(disc_generated_output, real_output, loss_func=None):
    if loss_func is None:
        loss_func = bce

    # Kaznjava se Diskriminator za svaku ne pogodjenu pravu sliku
    real_loss = loss_func(tf.ones_like(real_output), real_output)
    # Kaznjava se Diskriminator za svaku pogodjenu "laznu" sliku
    generated_loss = loss_func(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss
