import tensorflow as tf
import time
from IPython import display
from library.cyclegan.loss import discriminator_loss, generator_loss, cycle_loss, identity_loss
from library.cyclegan.visualization import test_image_generation
from library.cyclegan.loss import default_disc_alpha, default_gen_alpha, \
    default_identity_beta, default_cycle_alpha


class CycleGAN:
    def __init__(self, gen_g, gen_f,
                 disc_x, disc_y,
                 gen_g_optim, gen_f_optim,
                 disc_x_optim, disc_y_optim,
                 img_size,
                 disc_alpha=default_disc_alpha,
                 gen_alpha=default_gen_alpha,
                 cycle_alpha=default_cycle_alpha,
                 identity_beta=default_identity_beta,
                 summary_writer=None,
                 checkpoint=None,
                 checkpoint_prefix=None
                 ):
        self.gen_g = gen_g
        self.gen_f = gen_f
        self.disc_x = disc_x
        self.disc_y = disc_y

        self.gen_g_optim = gen_g_optim
        self.gen_f_optim = gen_f_optim
        self.disc_x_optim = disc_x_optim
        self.disc_y_optim = disc_y_optim

        self.img_size = img_size

        self.disc_alpha = disc_alpha
        self.gen_alpha = gen_alpha
        self.cycle_alpha = cycle_alpha
        self.identity_beta = identity_beta

        self.summary_writer = summary_writer
        self.checkpoint = checkpoint
        self.checkpoint_prefix = checkpoint_prefix

    @tf.function
    def train_step(self, real_images_x, real_images_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_images_y = self.gen_g(real_images_x)
            fake_images_x = self.gen_f(real_images_y)

            cycled_images_x = self.gen_f(fake_images_y)
            cycled_images_y = self.gen_g(fake_images_x)

            same_images_x = self.gen_f(real_images_x)
            same_images_y = self.gen_g(real_images_y)

            disc_real_images_x = self.disc_x(real_images_x)
            disc_real_images_y = self.disc_y(real_images_y)

            disc_fake_images_x = self.disc_x(fake_images_x)
            disc_fake_images_y = self.disc_y(fake_images_y)

            gen_g_loss = generator_loss(disc_fake_images_y, self.gen_alpha)
            gen_f_loss = generator_loss(disc_fake_images_x, self.gen_alpha)

            disc_x_loss = discriminator_loss(disc_real_images_x, disc_fake_images_x, self.disc_alpha)
            disc_y_loss = discriminator_loss(disc_real_images_y, disc_fake_images_y, self.disc_alpha)

            cycle_x_loss = cycle_loss(real_images_x, cycled_images_x, self.cycle_alpha)
            cycle_y_loss = cycle_loss(real_images_y, cycled_images_y, self.cycle_alpha)
            total_cycled_loss = cycle_x_loss + cycle_y_loss

            identity_x_loss = identity_loss(real_images_x, same_images_x, self.cycle_alpha, self.identity_beta)
            identity_y_loss = identity_loss(real_images_y, same_images_y, self.cycle_alpha, self.identity_beta)

            total_gen_g_loss = gen_g_loss + total_cycled_loss + identity_y_loss
            total_gen_f_loss = gen_f_loss + total_cycled_loss + identity_x_loss

        # Calculate the gradients for generator and discriminator
        gen_g_gradients = tape.gradient(total_gen_g_loss, self.gen_g.trainable_variables)
        gen_f_gradients = tape.gradient(total_gen_f_loss, self.gen_f.trainable_variables)

        disc_x_gradients = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
        disc_y_gradients = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.gen_g_optim.apply_gradients(zip(gen_g_gradients, self.gen_g.trainable_variables))
        self.gen_f_optim.apply_gradients(zip(gen_f_gradients, self.gen_f.trainable_variables))

        self.disc_x_optim.apply_gradients(zip(disc_x_gradients, self.disc_x.trainable_variables))
        self.disc_y_optim.apply_gradients(zip(disc_y_gradients, self.disc_y.trainable_variables))

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
        dot_period = test_generation_period // 10
        example_input, example_target = next(iter(val_ds.take(n_test_examples)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps + 1).enumerate():
            if (step + 1) % test_generation_period == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for {step + 1} steps: {time.time() - start:.2f} sec\n')

                test_image_generation(
                    self.gen_g, self.gen_f,
                    example_input, example_target,
                    rows=n_test_examples, img_size=self.img_size
                )
                print(f"Step: {step + 1}")

            # Training step
            self.train_step(input_image, target)

            if (step + 1) % dot_period == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 5k steps
            if self.checkpoint is not None and (step + 1) % checkpoint_period == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
