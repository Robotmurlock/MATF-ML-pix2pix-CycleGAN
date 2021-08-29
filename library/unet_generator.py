from tensorflow import keras
from tensorflow.keras import layers


def n_power2(n: int) -> int:
    """
    Pomocna funkcija izracunava stepen dvojke u broju.
    """
    result = 0
    while n % 2 == 0:
        result += 1
        n /= 2
    return result


def downsample(n_filters: int, kernel_size: int, apply_batchnorm: bool = True, name: str = None):
    """
    Pomocna funkcija za konstrukciju enkodera.

    Osnovna jedinica enkodera koja se koristi za Generator tj. U-Net arhitekturu i za Diskriminator.
    Sastoji se iz konvolutivnog sloja, opcionalnog sloja za unutrasnju standardizaciju i propustajuceg ReLU sloja.
    :param n_filters: Broj filtera konvolutivnog sloja mreze
    :param kernel_size: Dimenzija kernela
    :param apply_batchnorm: Da li se koristi unutrasnja standardizacija
    :param name: Ime modela (za vizualizaciju arhitekture)
    :return: Jednoslojni model: Izlaz je duplo manji od ulaza zbog koraka dimenzije 2.
    """
    model = keras.Sequential(name=name)
    model.add(layers.Conv2D(n_filters, kernel_size, strides=2, padding='same', use_bias=False))
    if apply_batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    return model


class UNetGeneratorBuilder:
    """
    Pomocna klasa za Konstrukciju generatora sa U-NET arhitekturom.

    :param input_size: Dimenzija slike ulaza.
    :param output_channels: Broj kanala generisane slike
    """

    def __init__(self, input_size: int, output_channels: int):
        self.input_size = input_size
        self.output_channels = output_channels

        self._encoder_layers = []
        self._encoder_layer_next_id = 1

        self._decoder_layers = []
        self._decoder_layer_next_id = 1

        self._locked = False

    def add_downsample_layer(self, n_filters: int, kernel_size: int, apply_batchnorm: bool = True):
        """
        Okvir za "downsample" funkciju koji dodeljuje odgovarajuce ime sloju.
        """
        self._encoder_layers.append(
            downsample(n_filters, kernel_size, apply_batchnorm, f'downsample{self._encoder_layer_next_id}'))
        self._encoder_layer_next_id += 1

    def add_upsample_layer(self, n_filters: int, kernel_size: int, apply_dropout: bool = False):
        """
        Analogno "add_downsample_layer" za dekoder.
        :param n_filters: Broj filtera transponovanog konvolutivnog sloja mreze
        :param kernel_size: Dimenzija kernela
        :param apply_dropout: Da li se koristi izostavljanje
        :return: Jednoslojni model: Izlaz je duplo veci od ulaza zbog koraka dimenzije 2.
        """
        model = keras.Sequential(name=f'upsample{self._decoder_layer_next_id}')
        self._decoder_layer_next_id += 1

        model.add(layers.Conv2DTranspose(n_filters, kernel_size, strides=2, padding='same', use_bias=False))
        if apply_dropout:
            model.add(layers.Dropout(0.5))
        model.add(layers.LeakyReLU())
        self._decoder_layers.append(model)

    def build(self):
        """
        Koraci generisanja modela:
        1. Provera se da li definisani slojevi enkodora i dekodera imaju smisla
            (da li ispunjavaju uslove za konstrukciju)
        2. Formira model sa spajanjem suprotnih veza
          Prvi sloj enkodera se spaja sa poslednjim slojem dekodera
          Drug sloj enkodera se spaja sa pretposlednjim sloje dekodera
          ...
          Poslednji sloj enkodera se spaja direktno sa prvim slojem dekodera, pa ne treba dodatno spajanje.
        :return: Generator sa U-Net arhitekturom
        """
        assert len(self._decoder_layers) + 1 == len(self._encoder_layers)
        assert len(self._decoder_layers) <= n_power2(self.input_size)

        inputs = layers.Input(shape=[self.input_size, self.input_size, 1], name='input')
        x = inputs

        # Encoder
        skips = []
        for down in self._encoder_layers:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Decoder (and skips)
        for up, skip in zip(self._decoder_layers, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        # Sa sigmoid aktivacionom funkcijom preslikavamo izlaz na [0, 1] interval
        final_layer = layers.Conv2DTranspose(self.output_channels, 3, strides=2, padding='same', activation='sigmoid')
        x = final_layer(x)

        return keras.Model(inputs=inputs, outputs=x)
