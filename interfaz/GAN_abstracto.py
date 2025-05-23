import os
import tensorflow as tf
from keras import layers, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import AdamW
import numpy as np
import matplotlib.pyplot as plt
import glob

#Forzamos GPU con 5GB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
            )
    except RuntimeError as e:
        print(e)

#####################################################
# Función para crear dataset con ImageDataGenerator #
#####################################################

def create_dataset_with_augmentation(damaged_dir, original_dir, batch_size=4):

    # Definimos ImageDataGenerator con data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect'
    )
    
    seed_value = 42 # Seed fijo para que las transformaciones sean iguales en damaged y original
    
    damaged_gen = datagen.flow_from_directory(
        damaged_dir,
        target_size=(512,512),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        seed= seed_value
    )
    
    original_gen = datagen.flow_from_directory(
        original_dir,
        target_size=(512,512),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        seed= seed_value
    )

    # Convertimos ambos generadores en tf.data.Dataset (cada uno un tensor de tensorflow)
    damaged_ds = tf.data.Dataset.from_generator(
        lambda: damaged_gen,
        output_signature=tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32)
    )
    original_ds = tf.data.Dataset.from_generator(
        lambda: original_gen,
        output_signature=tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32)
    )

    # Zipeamos damaged - original
    dataset = tf.data.Dataset.zip((damaged_ds, original_ds))
    return dataset

###########################
# Autoencoder (Generador) #
###########################

def build_autoencoder(input_shape=(512, 512, 3)):
    # input shape 512x512 y 3 canales (RGB)
    input_img = layers.Input(shape=input_shape)

    # --- Encoder --- #
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 512x512x3 -> 256x256x32
    p1 = MaxPooling2D((2, 2), padding='same')(c1)                           

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1) # 256x256x32 -> 128x128x64        
    p2 = MaxPooling2D((2, 2), padding='same')(c2)                           

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2) # 128x128x64 -> 64x64x128  
    p3 = MaxPooling2D((2, 2), padding='same')(c3)                           

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3) # 64x64x128 -> 32x32x256        
    p4 = MaxPooling2D((2, 2), padding='same')(c4)                           

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4) # 32x32x256 -> 16x16x512        
    p5 = MaxPooling2D((2, 2), padding='same')(c5)                           

    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(p5) # 16x16x512 -> 8x8x512        
    p6 = MaxPooling2D((2, 2), padding='same')(c6)                           

    c7 = Conv2D(512, (3, 3), activation='relu', padding='same')(p6) # 8x8x512 -> 4x4x512       
    p7 = MaxPooling2D((2, 2), padding='same')(c7)                           

    # Bottleneck (4x4)
    encoded = p7

    # --- Decoder (arquitectura U-net) --- #
    u1 = UpSampling2D((2,2))(encoded) # 4x4x512 -> 8x8x512
    d1 = Conv2D(512, (3,3), activation='relu', padding='same')(u1)
    m1 = concatenate([d1, c7])

    u2 = UpSampling2D((2,2))(m1) # 8x8x512 -> 16x16x512
    d2 = Conv2D(512, (3,3), activation='relu', padding='same')(u2)
    m2 = concatenate([d2, c6])

    u3 = UpSampling2D((2,2))(m2) # 16x16x512 -> 32x32x256
    d3 = Conv2D(512, (3,3), activation='relu', padding='same')(u3)
    m3 = concatenate([d3, c5])

    u4 = UpSampling2D((2,2))(m3) # 32x32x256 -> 64x64x128
    d4 = Conv2D(256, (3,3), activation='relu', padding='same')(u4)
    m4 = concatenate([d4, c4])

    u5 = UpSampling2D((2,2))(m4) # 64x64x128 -> 128x128x64
    d5 = Conv2D(128, (3,3), activation='relu', padding='same')(u5)
    m5 = concatenate([d5, c3])

    u6 = UpSampling2D((2,2))(m5) # 128x128x64 -> 256x256x32
    d6 = Conv2D(64, (3,3), activation='relu', padding='same')(u6)
    m6 = concatenate([d6, c2])

    u7 = UpSampling2D((2,2))(m6) # 256x256x32 -> 512x512x3
    d7 = Conv2D(32, (3,3), activation='relu', padding='same')(u7)
    m7 = concatenate([d7, c1])

    decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(m7)

    return Model(input_img, decoded)

############################
# Discriminador (PatchGAN) #
############################

def build_discriminator(input_shape=(512,512,3)):
    
    # input shape 512x512 y 3 canales (RGB)    
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (4,4), strides=2, padding='same')(inp) #512x512x3 -> 256x256x64
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, (4,4), strides=2, padding='same')(x) # 256x256x64 -> 128x128x128
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(256, (4,4), strides=2, padding='same')(x) # 128x128x128 -> 64x64x256
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(1, (4,4), strides=1, padding='same', activation='sigmoid')(x) # usamos sigmoid para convertir la salida a valores [0,1]

    return Model(inp, x)

#######################################
# Entrenamiento GAN manual
#######################################

def gan_train_loop(generator, discriminator, train_dataset, epochs=10, recon_weight=10.0):
    
    # Definimos optimizadores para el discriminador y generador
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    g_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    
    # Definimos la función loss BCE
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    # Listas para guardar los loss
    d_loss_list = []
    g_loss_list = []

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        steps=0
        
        # Iteramos sobre la tupla y el contador de step con enumerate
        for step, (damaged_imgs, real_imgs) in enumerate(train_dataset):
            steps+=1
            batch_size = tf.shape(damaged_imgs)[0]

            # Paso 1: entrenar discriminador
            # Usamos GradientTape para poder calcular los gradientes
            with tf.GradientTape() as d_tape:
                fake_imgs = generator(damaged_imgs, training=True) # Generamos una imagen falsa

                # (a) Reales
                real_pred = discriminator(real_imgs, training=True) # Predicción del discriminador (D(X)) para una imagen real (debería acercarse a 1)
                real_loss = bce(tf.ones_like(real_pred), real_pred) # Loss de la predicción: comparamos con un tensor de 1
                
                # (b) Falsas
                fake_pred = discriminator(fake_imgs, training=True) # Predicción del discriminador para una imagen falsa (debería acercarse a 0)
                fake_loss = bce(tf.zeros_like(fake_pred), fake_pred) # Loss de la predicción: comparamos con un tensor de 0

                # Accuracy del discriminador: queremos ≈ 0.5 en entrenamiento estable
                real_correct = tf.cast(real_pred > 0.5, tf.float32)   # 1 si clasifica real correctamente
                fake_correct = tf.cast(fake_pred <= 0.5, tf.float32)  # 1 si clasifica fake correctamente
                acc = 0.5 * (tf.reduce_mean(real_correct) + tf.reduce_mean(fake_correct))

                d_loss = 0.5 * (real_loss + fake_loss) # Promediamos el loss de la imagen real y el loss de la imagen falsa
                epoch_d_loss+=d_loss
            # Calculamos la lista de gradientes en base al d_loss de antes y aplicamos a todas las variables entrenables (weights)
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            
            # Paso 2: entrenar generador (autoencoder)
            with tf.GradientTape() as g_tape:
                fake_imgs = generator(damaged_imgs, training=True) # Generamos imagen falsa (G(Z))
                # "Congelamos" el discriminador (training=False) y sacamos una predicción
                fake_pred = discriminator(fake_imgs, training=False)

                # Pérdida adversarial: queremos que el discriminador diga "1" a fakes
                adv_loss = bce(tf.ones_like(fake_pred), fake_pred)
                
                #recon_loss = tf.reduce_mean(tf.abs(fake_imgs - real_imgs))
                # Pérdida de reconstrucción con SSIM (0 = perfecta, 1 = peor)
                ssim_vals    = tf.image.ssim(real_imgs, fake_imgs, max_val=1.0)
                recon_loss   = (1.0 - ssim_vals) / 2.0 
                recon_loss   = tf.reduce_mean(recon_loss)  # escalar medio por batch

                # Calculamos el loss dándole recon_weight = 10 "importancia" a la reconstrucción frente a engañar al discriminador
                g_loss = adv_loss + recon_weight * recon_loss
                epoch_g_loss+=g_loss

            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
            

            if step % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Step {step}] d_loss={d_loss.numpy():.4f} g_loss={g_loss.numpy():.4f} (adv={adv_loss.numpy():.4f}, rec={recon_loss.numpy():.4f}, acc={acc.numpy():.4f})")
        epoch_d_loss = epoch_d_loss/steps
        epoch_g_loss = epoch_g_loss/steps
        d_loss_list.append(epoch_d_loss)
        g_loss_list.append(epoch_g_loss)
        # Fin epoch
        print(f"--- Fin de epoch {epoch}, avg d_loss={epoch_d_loss.numpy():.4f}, avg g_loss={epoch_g_loss.numpy():.4f} ---")
    return d_loss_list, g_loss_list

#######################################
# Visualización de los resultados
#######################################

def plot_results(generator, dataset, n=2, epoch=None):
    for damaged_batch, real_batch in dataset.take(1):
        fake_batch = generator(damaged_batch, training=False)
        fake_batch = tf.clip_by_value(fake_batch, 0.0, 1.0)

        plt.figure(figsize=(3*n, 4))
        if epoch is not None:
            plt.suptitle(f"Epoch {epoch}")

        for i in range(n):
            # Dañado
            ax = plt.subplot(3, n, i+1)
            plt.imshow(damaged_batch[i].numpy())
            plt.title("Dañada")
            plt.axis("off")

            # Real
            ax = plt.subplot(3, n, i+1+n)
            plt.imshow(real_batch[i].numpy())
            plt.title("Original")
            plt.axis("off")

            # Fake
            ax = plt.subplot(3, n, i+1+2*n)
            plt.imshow(fake_batch[i].numpy())
            plt.title("Reconstruido")
            plt.axis("off")

        plt.show()
        break

#######################################
# MAIN
#######################################

def main():
    # Rutas
    train_dir_damaged = "/home/albadelatorre/Escritorio/abstract-art/damaged"
    train_dir_original = "/home/albadelatorre/Escritorio/abstract-art/original"

    # Creamos dataset
    train_dataset= create_dataset_with_augmentation(
        train_dir_damaged, train_dir_original, batch_size=2
    )

    # Construimos generador (autoencoder) y discriminador
    generator = build_autoencoder((512,512,3))
    discriminator = build_discriminator((512,512,3))
    
    # Definimos batch size
    train_dataset = train_dataset.take(1000)
    
    # Entrenamiento GAN
    g_list, d_list = gan_train_loop(
        generator=generator,
        discriminator=discriminator,
        train_dataset=train_dataset,
        epochs=100,
        recon_weight=10.0
    )
    
    # Visualizar ejemplos
    plot_results(generator, train_dataset, n=2)

    # Plot de las pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(g_list, label="Generator Loss")
    plt.plot(d_list, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.show()

    # Guardar generador (autoencoder) final
    generator.save("gan_abstracto.h5")

if __name__ == "__main__":
    main() 