import matplotlib.pyplot as plt
import tensorflow as tf


def show_image(img: tf.Tensor, title=None, img_size=None) -> None:
    """
    Pomocna funkcija za prikazivanja slika

    :param img: Slika koja treba da se prikaze
    :param title: Naslov koji ide uz sliku (opciono)
    :param img_size: Dimenzija slike (opciono)
    :return: None
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    if img_size is not None:
        img = tf.reshape(img, shape=(img_size, img_size))
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    plt.show()


def test_image_generation(gen_model, input_batch, target_batch, img_size, rows=4):
    """
    Pomocna funkcija za vizualizaciju perfomansi modela.
    :param gen_model: Generator
    :param input_batch: Ulaz (slike sa postavljenim kvadratom)
    :param target_batch: Izlaz (prave slike)
    :param img_size: Dimenzija slike koja se prikazuje
    :param rows: Broj primera koji se prikazuju
    :return: None
    """
    assert rows <= input_batch.shape[0]
    prediction_batch = gen_model(input_batch, training=True)

    fig, axs = plt.subplots(figsize=(18, 3 * rows), ncols=3, nrows=rows)
    for i in range(rows):
        p_img = prediction_batch[i]
        i_img = input_batch[i]
        t_img = target_batch[i]
        for j, (img, title) in enumerate([(i_img, 'Input'), (t_img, 'Target'), (p_img, 'Prediction')]):
            axs[i][j].imshow(tf.reshape(img, shape=(img_size, img_size)), cmap='gray')
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
    gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = zip(*history)
    axs[0][col].plot(gen_total_loss, color='red', label='Total Loss')
    axs[0][col].plot(gen_gan_loss, color='blue', label='GEN Loss')
    axs[0][col].plot(gen_l1_loss, color='green', label='L1 Loss')
    axs[0][col].set_title(f'{prefix} GAN Loss')
    axs[0][col].set_xlabel('Step')
    axs[0][col].set_ylabel('Loss')
    axs[0][col].legend()

    axs[1][col].plot(disc_loss, color='red')
    axs[1][col].set_title(f'{prefix} DISC Loss')
    axs[1][col].set_xlabel('Step')
    axs[1][col].set_ylabel('Loss')

    axs[2][col].plot(disc_loss, color='red', label='DISC Loss')
    axs[2][col].plot(gen_gan_loss, color='blue', label='GEN Loss')
    axs[2][col].set_title(f'{prefix} DISC vs GEN Loss')
    axs[2][col].set_xlabel('Step')
    axs[2][col].set_ylabel('Loss')
    axs[2][col].legend()


def plot_training_results(train_history, val_history):
    """
    Prikazuje grafike koji predstavljaju rezultate treniranja nad skupom za ucenje i nad skupom za validaciju.
    :param train_history: Istorija funkcije greske nad skupom za ucenje
    :param val_history: Istorija funkcije greske nad skupom za validaciju
    """
    fig, axs = plt.subplots(figsize=(12, 10), nrows=3, ncols=2)
    plot_history(train_history, axs, col=0, prefix='Train:')
    plot_history(val_history, axs, col=1, prefix='Validation:')
    plt.tight_layout()
