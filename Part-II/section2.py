
import tensorflow as tf
import time

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from utils import generateDataset, splitDataset, plotPredictios, plotWeights, plotComparePredictions


######################################
# Parameters
section = "4.3.2.4"               # Section 4.3.1 or 4.3.2
start_t = 301                   # The dataset will be t in {start_t : end_t}
end_t = 1500
n_epochs = 200                  # number of complete pass in the dataset for training
patience = 3                    # patience for the early stopping
l1_reg_const = 0.001            # const for l1 regularization penalty
l2_reg_const = 0.001            # const for l2 regularization penalty
sigma_noise = 0.09              # For the noise in 4.3.2

######################################

## noise    reg_const      mse
#  0.18        0.05       0.0887
#  0.18        0.01       0.0794
#  0.18        0.001      0.0630
#  0.18         0         0.0662

#  0.09        0.05       0.0617
#  0.09        0.01       0.0264
#  0.09        0.001      0.0208
#  0.09         0         0.0221

#  0.03        0.05       0.0438
#  0.03        0.01       0.0136
#  0.03        0.001      0.0045
#  0.03         0         0.0062


if section == "4.3.1":
    input, output = generateDataset(start_t, end_t)

    x_train , x_val , x_test, y_train, y_val, y_test = splitDataset(input, output)

    # Model
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])

    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    callback = EarlyStopping(monitor='val_loss', patience=patience)

    model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
              validation_data=(x_val,  y_val))

    model.evaluate(x_val,  y_val, verbose=2)

    y_pred = model.predict(x_test)

    plotPredictios(y_pred, y_test, len(input))

    plotWeights(model, l1_reg_const)


if section == "4.3.2":
    input, output = generateDataset(start_t, end_t, noise=True, sigma_noise=sigma_noise)

    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    # Model
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])

    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    callback = EarlyStopping(monitor='val_loss', patience=patience)

    model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
              validation_data=(x_val,  y_val))

    model.evaluate(x_val, y_val, verbose=2)

    y_pred = model.predict(x_test)

    plotPredictios(y_pred, y_test, len(input))

    plotWeights(model, l1_reg_const)


if section == "4.3.2.3":
    input, output = generateDataset(start_t, end_t, noise=True, sigma_noise=sigma_noise)

    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    # Model
    # Model
    model1 = tf.keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model2 = tf.keras.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model1.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])

    model2.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])

    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    callback = EarlyStopping(monitor='val_loss', patience=patience)

    model1.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
              validation_data=(x_val,  y_val))

    model2.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
              validation_data=(x_val,  y_val))

    model1.evaluate(x_val, y_val, verbose=2)
    model2.evaluate(x_val, y_val, verbose=2)

    y_pred_1 = model1.predict(x_test)  # mse: 0.0208
    y_pred_2 = model2.predict(x_test)  # mse: 0.0244

    plotComparePredictions(y_pred_1, y_pred_2, y_test, len(input), "two layers", "three layers")


if section == "4.3.2.4":
    input, output = generateDataset(start_t, end_t, noise=True, sigma_noise=sigma_noise)

    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    # Model
    # Model
    model1 = tf.keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model2 = tf.keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model1.compile(optimizer='adam',
                   loss='mse',
                   metrics=['mse'])

    model2.compile(optimizer='adam',
                   loss='mse',
                   metrics=['mse'])

    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    callback = EarlyStopping(monitor='val_loss', patience=patience)


    start1 = time.time()
    model1.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
               validation_data=(x_val, y_val))
    end1 = time.time()


    start2 = time.time()
    model2.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
               validation_data=(x_val, y_val))
    end2 = time.time()
    print("Time to train model1: ", end1 - start1)
    print("Time to train model2: ", end2 - start2)
