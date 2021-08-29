import matplotlib.pyplot as plt
import tensorflow as tf


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
