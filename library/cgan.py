import tensorflow as tf
from IPython import display
import time
from library.visualization import test_image_generation
from library.loss import generator_loss, discriminator_loss


class CGAN:
    def __init__(self,
                 generator,
                 discriminator,
                 alpha,
                 generator_optimizer,
                 discriminator_optimizer,
                 img_size,
                 summary_writer=None,
                 checkpoint=None,
                 checkpoint_prefix=None
                 ):
        """
        :param generator: Generator model
        :param discriminator: Diskriminator model
        :param alpha: Parametar za odnos izmedju GAN i L1 greske kod generator modela
        :param generator_optimizer: Optimizator za generator
        :param discriminator_optimizer: Optimizator za diskriminator
        :param summary_writer: Pomocni objekat za logovanje
        :param checkpoint: Pomocni objekat za cuvanje modela
        :param checkpoint_prefix: Putanja gde se cuva "checkpoint"
        """
        self.generator = generator
        self.discriminator = discriminator
        self.alpha = alpha
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.img_size = img_size

        self.summary_writer = summary_writer
        self.checkpoint = checkpoint
        self.checkpoint_prefix = checkpoint_prefix
        if self.checkpoint_prefix is None:
            self.checkpoint_prefix = '.'

    @tf.function
    def train_step(self, input_image, target, step):
        """
        Za svaki korak ucenja se vrsi vise operacija:
        1. Generator generise izlaz na osnovu ulaza
        2. Diskriminator ocenjuje stvarne slike (da li su prave ili ne)
        3. Diskriminator ocenjuje generisane slike (da li su prave ili ne)
        4. Odredjuje se greska Generatora na osnovu ocene Diskriminatora (da li je prevara uspesna) iz koraka #3
          i na osnovu stvarne slike (L1 greska)
        5. Odredjuje se greska Diskriminatora na osnovu ocena iz koraka #2 i #3

        :param input_image: Ulaz generatora
        :param target: Prava slika
        :param step: Broj trenutnog koraka
        :return: Loss
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 1
            gen_output = self.generator(input_image, training=True)
            # 2
            disc_real_output = self.discriminator([input_image, target], training=True)
            # 3
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            # 4
            gen_total_loss, gen_gan_loss, gen_l1_loss = \
                generator_loss(disc_generated_output, gen_output, target, alpha=self.alpha)
            # 5
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
                tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def eval_step(self, input_image, target):
        """
        Evaluira rezultate za dati ulaz
        :param input_image: Ulaz generatora
        :param target: Prava slika
        :return: Loss
        """
        gen_output = self.generator(input_image)
        disc_real_output = self.discriminator([input_image, target])
        disc_generated_output = self.discriminator([input_image, gen_output])
        gen_total_loss, gen_gan_loss, gen_l1_loss = \
            generator_loss(disc_generated_output, gen_output, target, alpha=self.alpha)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def fit(self,
            train_ds,
            val_ds,
            steps,
            test_generation_period=1000,
            n_test_examples=4,
            checkpoint_period=5000):
        """
        Napomena: Namera je da se funkcija koristi u okviru IPython sveske

        Proces ucenja zasnovan na funkciji "train_step" koji dodatno:
        - prikazuje rezultate;
        - ispisuje stanje ucenja;
        - Kreira periodicno checkpoint.

        :param train_ds: Skup za treniranje
        :param val_ds: Skup za validaciju
        :param steps: Broj koraka treniranja
        :param test_generation_period: Broj koraka izmedju vizualizacije novih rezultata
        :param n_test_examples: Broj instanci koje se vizualizuju
        :param checkpoint_period: Period za cuvanje rezultata
        :return: Loss History
        """
        train_history, val_history = [], []
        dot_period = test_generation_period // 10

        example_input, example_target = next(iter(val_ds.take(n_test_examples)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps+1).enumerate():
            if (step+1) % test_generation_period == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for {step+1} steps: {time.time()-start:.2f} sec\n')

                test_image_generation(
                    self.generator, example_input, example_target, rows=n_test_examples, img_size=self.img_size)
                print(f"Step: {step+1}")

                for val_input_image, val_target in val_ds:
                    val_history.append(self.eval_step(val_input_image, val_target))

            train_history.append(self.train_step(input_image, target, step))

            # Training step
            if (step+1) % dot_period == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 5k steps
            if self.checkpoint is not None and (step + 1) % checkpoint_period == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        return train_history, val_history
