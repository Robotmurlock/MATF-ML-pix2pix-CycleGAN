import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np


def load_data(dataset_name: str):
    """
    Pomocna funkcija za ucitavanje MNIST skupa podataka
    :param dataset_name: Ime skupa podataka. U opticaju su
        - 'mnist'
        - 'fashion_mnist'
    :return: Skup podataka za ucenje i testiranje
    """
    assert dataset_name in ['mnist', 'fashion_mnist']
    (tf_train, tf_test), ds_info = tfds.load(
        dataset_name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=False,
        with_info=True,
    )
    return tf_train, tf_test


# Obrada podataka
def create_mnist_input(img, radius=4):
    """
    Posto su tenzori imutabilni (Osim ako se ne koristi Variable),
    u ovom slucaju se slika deli na 5 delova u odnosu na deo slike koji se brise
    1. Deo slike koji se brise tj. postavlja se sivi kvadrat
    2. Deo slike levo od kvadrata
    3. Deo slike desno od kvadrata
    4. Deo slike iznad kvadrata
    5. Deo slike ispod kvadrata
    2 2 2 4 4 3
    2 2 2 4 4 3
    2 2 2 4 4 3
    2 2 2 1 1 3
    2 2 2 1 1 3
    2 2 2 4 4 3

    Kvadrat se ne postavlja na ivicama kako bi zadatak bio nesto tezi
    (model bi lako naucio da ivice uvek popuni crnom bojom jer se tu ne nalaze brojevi).

    Alternativa bi bila da se podaci pretprocesiraju koriscenjem numpy.

    :param img: Slika
    :param radius: Radijus kvadrata, polovina stranice
    :return: Slika sa kvadratom
    """
    img_input = tf.identity(img)
    x_coord = np.random.randint(max(radius, 8), min(img.shape[0]-8, img.shape[1]-radius))
    y_coord = np.random.randint(max(radius, 8), min(img.shape[0]-8, img.shape[1]-radius))

    img_input_left = img_input[:x_coord-radius, :, :]
    img_input_right = img_input[x_coord+radius:, :, :]
    img_input_top = img_input[x_coord-radius:x_coord+radius, :y_coord-radius, :]
    img_input_bottom = img_input[x_coord-radius:x_coord+radius, y_coord+radius:, :]
    img_input_center = tf.ones(shape=(2*radius, 2*radius, 1)) * 64
    img_input_bottom_center_top = tf.concat([img_input_top, img_input_center, img_input_bottom], axis=1)
    img_input = tf.concat([img_input_left, img_input_bottom_center_top, img_input_right], axis=0)

    return img_input, img


def tf_pipeline(tf_dataset, img_size: int, batch_size: int):
    """
    Funkcija za pripremu MNIST skupa podataka.
    Skup se prvenstveno konvertuje u odgovarajuci oblik,
    onda se dodaje sum na podatke (pogledati rad), i na
    kraju se skup priprema za obradu (skaliranje, mesanje i podela na batch-eve)

    :param tf_dataset: Skup podataka
    :param img_size: Dimenzija slike (odnosi se na izlaz)
    :param batch_size: Dimenzija batch-a
    :return: Pripremljen skup podataka
    """
    # noinspection PyUnresolvedReferences
    rotation = layers.RandomRotation(factor=(-0.1, 0.1))
    flip = tf.keras.layers.RandomFlip()

    # Konvertovanje ulaza u sliku dimenzije [img_size]x[img_size]
    tf_dataset = tf_dataset.map(lambda x: tf.cast(x['image'], tf.float32))
    tf_dataset = tf_dataset.map(lambda x: tf.image.resize(x, (img_size, img_size), method='nearest'))

    # Neophodno je da postoji neka vrsta suma
    tf_dataset = tf_dataset.map(lambda x: rotation(x))
    tf_dataset = tf_dataset.map(lambda x: x if np.random.random() <= 0.5 else flip(x))

    # Dodavanje nasumicne "beline" na sliku
    tf_dataset = tf_dataset.map(create_mnist_input)

    # Pikseli sa normalizuju na interval [0, 1]
    tf_dataset = tf_dataset.map(lambda x, y: (x / 255.0, y / 255.0))
    tf_dataset = tf_dataset.shuffle(100)
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset


def prepare_mnist_dataset(dataset_name: str, img_size: int = 32, batch_size: int = 40):
    """
    Ucitavavanje i priprema MNIST skupa podataka.

    :param dataset_name: Ime skupa podataka ('fashion_mnist' ili 'mnist')
    :param img_size: Dimenzija slike u podacima
    :param batch_size: Dimenzija 'Batch'-a
    :return: Skup podataka za ucenje i skup podataka za testiranje
    """
    tf_train, tf_test = load_data(dataset_name)
    tf_train = tf_pipeline(tf_train, img_size, batch_size)
    tf_test = tf_pipeline(tf_test, img_size, batch_size)
    return tf_train, tf_test
