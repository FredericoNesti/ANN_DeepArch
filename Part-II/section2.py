import time
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from collections import defaultdict

from utils import generateDataset, plotTimeSeries, splitDataset, plotPredictios, plotWeights, plotNoisyPredictions


######################################
# Parameters
section = "4.3.2.4"             # Section 4.3.1.1 , 4.3.1.2 ... 4.3.2.4
start_t = 301                   # The dataset will be t in {start_t : end_t}
end_t = 1500
n_epochs = 1000                 # number of complete pass in the dataset for training
patience = 3                    # patience for the early stopping
l1_reg_const = 0.001            # const for l1 regularization penalty
l2_reg_const = 0.001            # const for l2 regularization penalty

######################################

if section == "4.3.1.1":
    _, output = generateDataset(start_t, end_t)
    t = [i for i in range(start_t, end_t)]
    plotTimeSeries(t, output)

if section == "4.3.1.2":
    input, output = generateDataset(start_t, end_t)
    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    sizes = [8, 16, 32, 64, 128, 256, 512]  # models of NN to be trained
    nn2error = defaultdict(list)

    # Model
    for size in tqdm(sizes):  # for each model size
        for i in range(5):  # train 5 times for statistical significance
            model = tf.keras.Sequential([
                layers.Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                             input_shape=[len(x_train[0])]),
                layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mse'])

            # This callback will stop the training when there is no improvement in
            # the validation loss for three consecutive epochs.
            callback = EarlyStopping(monitor='val_loss', patience=patience)

            model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val,  y_val), verbose=0)

            nn2error[size].append(model.evaluate(x_val,  y_val, verbose=0)[1])
    print("##########################################################")
    print("Evaluation result")
    for size,errors in nn2error.items():
        print("Model: {}x1     mse: {:.5f} +- {:.5f}".format(size, np.mean(errors), np.std(errors)))
    print("##########################################################")

if section == "4.3.1.3":
    input, output = generateDataset(start_t, end_t)
    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    reg_cons = [(0.0,0.0), (0.001, 0.001), (0.05,0.05), (0.1,0.1)]  # models of NN to be trained
    reg_cons2error = defaultdict(list)

    # Model
    for l1, l2 in tqdm(reg_cons):
        for i in range(5):
            model = tf.keras.Sequential([
                layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2),
                             input_shape=[len(x_train[0])]),
                layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mse'])

            # This callback will stop the training when there is no improvement in
            # the validation loss for three consecutive epochs.
            callback = EarlyStopping(monitor='val_loss', patience=patience)

            model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val,  y_val), verbose=0)

            reg_cons2error[(l1,l2)].append(model.evaluate(x_val,  y_val, verbose=0)[1])

    print("##########################################################")
    print("Effect of regularization")
    for (l1,l2), errors in reg_cons2error.items():
        print("Model: 32x1  reg_cons: {},{}   mse: {:.5f} +- {:.5f}".format(l1, l2, np.mean(errors), np.std(errors)))
    print("##########################################################")

    # Effect of regularization on weights
    reg_cons = [(0.0,0.0), (0.01,0.01), (0.1,0.1)]
    for (l1, l2) in tqdm(reg_cons):

        model = tf.keras.Sequential([
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2),
                         input_shape=[len(x_train[0])]),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mse'])

        # This callback will stop the training when there is no improvement in
        # the validation loss for three consecutive epochs.
        callback = EarlyStopping(monitor='val_loss', patience=patience)

        model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val, y_val), verbose=0)

        y_pred = model.predict(x_test)
        plotPredictios(y_pred, y_test, len(input))
        plotWeights(model, l1)


if section == "4.3.1.4":
    input, output = generateDataset(start_t, end_t)
    x_train , x_val , x_test, y_train, y_val, y_test = splitDataset(input, output)

    # Model
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    callback = EarlyStopping(monitor='val_loss', patience=patience)

    model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val,  y_val), verbose=0)

    y_pred = model.predict(x_test)

    ## Our model
    # our_model = OURMOODEL()
    # our_model.fit(x_train, y_train)
    # our_y_pred = our_model.predict(x_test)

    # plotComparePredictions(y_pred, our_y_pred, y_test, n_input=len(input), name_model1="TF 64x1 model",
    #                        name_model2="Our model")

if section == "4.3.2.1":

    sigma_noise = [0.03, 0.09, 0.18]
    third_layers_size = [16, 32, 64, 128, 256]
    noise_model2error = defaultdict(list)

    for sn in tqdm(sigma_noise):
        input, output = generateDataset(start_t, end_t, noise=True, sigma_noise=sn)
        x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

        for layer_size in third_layers_size:
            for i in range(5):  # run 5 times to statistical purpose
                # Model
                model = tf.keras.Sequential([
                    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                                 input_shape=[len(x_train[0])]),
                    layers.Dense(layer_size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
                    layers.Dense(1)
                ])

                model.compile(optimizer='adam', loss='mse', metrics=['mse'])

                # This callback will stop the training when there is no improvement in
                # the validation loss for three consecutive epochs.
                callback = EarlyStopping(monitor='val_loss', patience=patience)

                model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val,  y_val), verbose=0)

                noise_model2error[(sn,layer_size)].append(model.evaluate(x_val, y_val, verbose=0)[1])

    print("##########################################################")
    print("Evaluation result")
    for (noise, layer_size), errors in noise_model2error.items():
        print("Model: 32x{}x1  noise: {}   mse: {:.5f} +- {:.5f}".format(layer_size, noise, np.mean(errors),
                                                                         np.std(errors)))
    print("##########################################################")

if section == "4.3.2.2":
    reg_cons = [(0.0,0.0), (0.001, 0.001), (0.05,0.05), (0.1,0.1)]
    sigma_noise = [0.03, 0.09, 0.18]
    noise_reg2error = defaultdict(list)

    # Best model found on 4.3.2.1 : 32x64
    for sn in tqdm(sigma_noise):
        input, output = generateDataset(start_t, end_t, noise=True, sigma_noise=sn)
        x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

        for l1, l2 in reg_cons:
            for i in range(5):
                model = tf.keras.Sequential([
                    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                                 input_shape=[len(x_train[0])]),
                    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
                    layers.Dense(1)
                ])

                model.compile(optimizer='adam', loss='mse', metrics=['mse'])

                # This callback will stop the training when there is no improvement in
                # the validation loss for three consecutive epochs.
                callback = EarlyStopping(monitor='val_loss', patience=patience)

                model.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val, y_val),
                          verbose=0)

                noise_reg2error[(sn,l1,l2)].append(model.evaluate(x_val, y_val, verbose=0)[1])

    print("##########################################################")
    print("Evaluation result section 4.3.2.2")
    for (noise, reg1, reg2), errors in noise_reg2error.items():
        print("Model: 32x64x1  noise: {}  reg_fact: {}  mse: {:.5f} +- {:.5f}".format(noise, reg1, np.mean(errors),
                                                                                      np.std(errors)))
    print("##########################################################")

if section == "4.3.2.3": # Compare 2-layers with 3-layers
    sm = 0.09
    input, output = generateDataset(start_t, end_t, noise=True, sigma_noise=sm)
    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    # Models
    model1 = tf.keras.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(1)
    ])

    model2 = tf.keras.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                     input_shape=[len(x_train[0])]),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const)),
        layers.Dense(1)
    ])

    model1.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model2.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    callback = EarlyStopping(monitor='val_loss', patience=patience)
    print("Model1 training...")
    model1.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val,  y_val), verbose=0)

    print("Model2 training...")
    model2.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback], validation_data=(x_val,  y_val), verbose=0)

    model1.evaluate(x_val, y_val, verbose=2)
    model2.evaluate(x_val, y_val, verbose=2)

    y_pred_1 = model1.predict(x_test)  # mse: 0.0208
    y_pred_2 = model2.predict(x_test)  # mse: 0.0244

    plotNoisyPredictions(1301, 1301 + len(y_test), y_pred_1, y_test, "2L")
    # plotNoisyPredictions(1301, 1301 + len(y_test), y_pred_2, y_test, "3L")
    # plotNoisyPredictions(1301, 1301 + len(y_test), y_pred_1, y_pred_2, y_test, len(input), "2L", "3L")


if section == "4.3.2.4":  # Time evaluation
    input, output = generateDataset(start_t, end_t, noise=False)
    x_train, x_val, x_test, y_train, y_val, y_test = splitDataset(input, output)

    n_epochs1, n_epochs2 = 0,0

    start1 = time.time()
    for i in tqdm(range(5)):
        model1 = tf.keras.Sequential([
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                         input_shape=[len(x_train[0])]),
            layers.Dense(1)
        ])
        model1.compile(optimizer='adam', loss='mse', metrics=['mse'])
        callback = EarlyStopping(monitor='val_loss', patience=patience)
        model1.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
                   validation_data=(x_val, y_val), verbose=0)
    end1 = time.time()

    start2 = time.time()
    for i in tqdm(range(5)):
        model2 = tf.keras.Sequential([
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1_reg_const, l2_reg_const),
                         input_shape=[len(x_train[0])]),
            layers.Dense(1)
        ])
        model2.compile(optimizer='adam', loss='mse', metrics=['mse'])
        callback = EarlyStopping(monitor='val_loss', patience=patience)
        model2.fit(x_train, y_train, epochs=n_epochs, callbacks=[callback],
                   validation_data=(x_val, y_val),verbose=0)
    end2 = time.time()

    print("Time to train model1: {:.5f}".format((end1 - start1)/5))
    print("Time to train model2: {:.5f}".format((end2 - start2)/5))
