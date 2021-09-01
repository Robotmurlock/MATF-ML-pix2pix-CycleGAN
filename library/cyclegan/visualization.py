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


def plot_history(history, axs, col, prefix=''):
    """
    :param history: Istorija funkcije greske
    :param axs: Grafice nad kojim se slika
    :param col: Kolona koja se popunjava (ako ima vise skupova, moze da se gleda uporedo rezultat)
    :param prefix: Prefiks za rezultat u izabranoj koloni
    """
    gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, total_cycled_loss, identity_x_loss, identity_y_loss = \
        zip(*history)

    axs[0][col].plot(gen_g_loss, color='red', label='Total GEN[G] Loss')
    axs[0][col].plot(gen_f_loss, color='blue', label='Total GEN[F] Loss')
    axs[0][col].set_title(f'{prefix} GEN Loss')
    axs[0][col].set_xlabel('Step')
    axs[0][col].set_ylabel('Loss')
    axs[0][col].legend()

    axs[1][col].plot(disc_y_loss, color='red', label='Total DISC[Y] loss')
    axs[1][col].plot(disc_x_loss, color='blue', label='Total DISC[X] loss')
    axs[1][col].set_title(f'{prefix} DISC Loss')
    axs[1][col].set_xlabel('Step')
    axs[1][col].set_ylabel('Loss')

    axs[2][col].plot(total_cycled_loss, color='red', label='Cycle Loss')
    axs[2][col].set_title(f'{prefix} Cycle Loss')
    axs[2][col].set_xlabel('Step')
    axs[2][col].set_ylabel('Loss')

    axs[3][col].plot(identity_x_loss, color='red', label='Total Identity[X] loss')
    axs[3][col].plot(identity_y_loss, color='blue', label='Total Identity[Y] loss')
    axs[3][col].set_title(f'{prefix} Identity Loss')
    axs[3][col].set_xlabel('Step')
    axs[3][col].set_ylabel('Loss')

    axs[2][col].legend()


def plot_training_results(train_history, val_history):
    """
    Prikazuje grafike koji predstavljaju rezultate treniranja nad skupom za ucenje i nad skupom za validaciju.
    :param train_history: Istorija funkcije greske nad skupom za ucenje
    :param val_history: Istorija funkcije greske nad skupom za validaciju
    """
    fig, axs = plt.subplots(figsize=(12, 10), nrows=4, ncols=2)
    plot_history(train_history, axs, col=0, prefix='Train:')
    plot_history(val_history, axs, col=1, prefix='Validation:')
    plt.tight_layout()
