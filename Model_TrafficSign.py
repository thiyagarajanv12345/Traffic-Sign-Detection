import numpy as np
import cv2 as cv
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, UpSampling2D, Concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from Evaluation_All import seg_evaluation


def TrafficSign_Seg(input_shape=(512, 512, 3), num_classes=1):
    inputs = Input(input_shape)

    # ---------------- Encoder ----------------
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)  # 256
    p1 = BatchNormalization()(x)

    x = Conv2D(128, 3, strides=2, activation='relu', padding='same')(p1)  # 128
    p2 = BatchNormalization()(x)

    x = Conv2D(256, 3, strides=2, activation='relu', padding='same')(p2)  # 64
    p3 = BatchNormalization()(x)

    x = Conv2D(512, 3, strides=2, activation='relu', padding='same')(p3)  # 32
    p4 = BatchNormalization()(x)

    # ---------------- Bottleneck ----------------
    b = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    b = Conv2D(1024, 3, activation='relu', padding='same')(b)

    # ---------------- Decoder ----------------
    u1 = UpSampling2D()(b)  # 64
    u1 = Concatenate()([u1, p3])
    u1 = Conv2D(512, 3, activation='relu', padding='same')(u1)

    u2 = UpSampling2D()(u1)  # 128
    u2 = Concatenate()([u2, p2])
    u2 = Conv2D(256, 3, activation='relu', padding='same')(u2)

    u3 = UpSampling2D()(u2)  # 256
    u3 = Concatenate()([u3, p1])
    u3 = Conv2D(128, 3, activation='relu', padding='same')(u3)

    u4 = UpSampling2D()(u3)  # 512
    u4 = Concatenate()([u4, inputs])
    u4 = Conv2D(64, 3, activation='relu', padding='same')(u4)

    # ---------------- Output ----------------
    out = Conv2D(num_classes, 1, activation='sigmoid', padding='same')(u4)

    model = Model(inputs, out, name="TrafficSign_Segmentation")
    return model


def Model_TrafficSign(Images, GroundTruth, Epoch):
    N = Images.shape[0]
    IMG_SIZE = 512

    X = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    Y = np.zeros((N, IMG_SIZE, IMG_SIZE, GroundTruth.shape[-1]), dtype=np.float32)

    # -------- Preprocessing --------
    for i in range(N):
        im = cv.resize(Images[i], (IMG_SIZE, IMG_SIZE))
        gt = cv.resize(GroundTruth[i], (IMG_SIZE, IMG_SIZE))

        X[i] = im / 255.0
        Y[i] = gt / 255.0

    num_classes = Y.shape[-1]

    model = TrafficSign_Seg(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=num_classes
    )

    model.summary()

    # -------- Compile --------
    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # -------- Callbacks --------
    earlystop = EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        'TrafficSign_best.keras',
        save_best_only=True,
        verbose=1
    )

    # -------- Train --------
    model.fit(
        X, Y,
        batch_size=4,
        epochs=Epoch,
        validation_split=0.1,
        callbacks=[earlystop, checkpoint]
    )

    # -------- Test --------
    split = int(0.9 * N)
    X_test = X[split:]
    Y_test = Y[split:]

    preds = model.predict(X_test, batch_size=4)

    # Evaluation
    Eval = seg_evaluation(preds, Y_test)

    return Eval, preds
