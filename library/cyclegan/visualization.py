import matplotlib.pyplot as plt
import tensorflow as tf


def test_image_generation(gen_g, gen_f, input_batch, target_batch, img_size, rows=3):
    """
    Pomocna funkcija za vizualizaciju perfomansi modela.
    :param gen_g: G Generator
    :param gen_f: F Generator
    :param input_batch: Ulaz (slike sa postavljenim kvadratom)
    :param target_batch: Izlaz (prave slike)
    :param img_size: Dimenzija slike koja se prikazuje
    :param rows: Broj primera koji se prikazuju
    :return: None
    """
    assert rows <= input_batch.shape[0]
    prediction_batch_g = gen_g(input_batch, training=True)
    prediction_batch_f = gen_f(target_batch, training=True)
    cycled_batch_gf = gen_f(prediction_batch_g, training=True)
    cycled_batch_fg = gen_g(prediction_batch_f, training=True)
    same_batch_g = gen_g(target_batch, training=True)
    same_batch_f = gen_f(input_batch, training=True)
    channels = input_batch.shape[-1]

    fig, axs = plt.subplots(figsize=(18, 3 * rows), ncols=8, nrows=rows)
    for i in range(rows):
        pg_img = prediction_batch_g[i]
        pf_img = prediction_batch_f[i]
        i_img = input_batch[i]
        t_img = target_batch[i]
        cg_img = cycled_batch_gf[i]
        cf_img = cycled_batch_fg[i]
        sg_img = same_batch_g[i]
        sf_img = same_batch_f[i]
        for j, (img, title) in enumerate([(i_img, 'Input'), (t_img, 'Target'),
                                          (pg_img, 'Prediction G'), (cg_img, 'Cycled X->Y->X'),
                                          (sg_img, 'Same G'), (pf_img, 'Prediction F'),
                                          (cf_img, 'Cycled Y->X->Y'), (sf_img, 'Same F'), ]):
            new_shape = img_size, img_size, channels
            if channels == 1:
                new_shape = new_shape[:-1]
            img_reshaped = tf.reshape(img, shape=new_shape)
            axs[i][j].imshow(img_reshaped, cmap='gray')
            axs[i][j].set_title(title)
            axs[i][j].axis('off')
    plt.show()
