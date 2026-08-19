import numpy as np
import cv2 as cv
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, UpSampling2D, Concatenate
from keras.optimizers import Adam
from Evaluation_All import seg_evaluation


# YOLOv8 Encoder–Decoder
def yolo_v8_seg(input_shape=(512, 512, 3), num_classes=3):
    inputs = Input(input_shape)

    # Encoder (YOLOv8 backbone-like)
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)     # 256x256
    p1 = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = Conv2D(128, 3, strides=2, activation='relu', padding='same')(p1)   # 128x128
    p2 = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = Conv2D(256, 3, strides=2, activation='relu', padding='same')(p2)   # 64x64
    p3 = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = Conv2D(512, 3, strides=2, activation='relu', padding='same')(p3)   # 32x32
    p4 = Conv2D(512, 3, activation='relu', padding='same')(x)

    # Bottleneck
    b = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    b = Conv2D(1024, 3, activation='relu', padding='same')(b)

    # Decoder (upsampling path)
    u1 = UpSampling2D((2, 2), interpolation='bilinear')(b)  # 64x64
    u1 = Concatenate()([u1, p3])
    u1 = Conv2D(512, 3, activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(u1)  # 128x128
    u2 = Concatenate()([u2, p2])
    u2 = Conv2D(256, 3, activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(u2)  # 256x256
    u3 = Concatenate()([u3, p1])
    u3 = Conv2D(128, 3, activation='relu', padding='same')(u3)

    u4 = UpSampling2D((2, 2), interpolation='bilinear')(u3)  # 512x512
    u4 = Concatenate()([u4, inputs])
    u4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    u4 = Conv2D(64, 3, activation='relu', padding='same')(u4)

    # Output layer
    out = Conv2D(num_classes, 1, activation='sigmoid', padding='same')(u4)

    model = Model(inputs=inputs, outputs=out, name='YOLOv8_Segmentation')
    return model


#  Training Function
def Model_YOLOv8(Images, Groundtruth, Epoch):
    N = Images.shape[0]

    IMG_SIZE = 512
    X = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    Y = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    # Resize and normalize
    for i in range(N):
        im = Images[i]
        gt = Groundtruth[i]
        imr = cv.resize(im, (IMG_SIZE, IMG_SIZE))
        gtr = cv.resize(gt, (IMG_SIZE, IMG_SIZE))
        X[i] = imr.astype(np.float32) / 255.0
        Y[i] = gtr.astype(np.float32) / 255.0

    num_classes = Y.shape[-1]
    model = yolo_v8_seg(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)
    model.summary()

    # Compile
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])

    # Train
    earlystopper = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
    checkpointer = ModelCheckpoint('yolov8_best.keras', save_best_only=True, verbose=1)

    model.fit(X, Y, validation_split=0.1, batch_size=4, epochs=Epoch,
              callbacks=[earlystopper, checkpointer])

    # Test set
    split_idx = int(0.9 * N)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    preds = model.predict(X_test, batch_size=4)

    # Evaluation
    Eval = seg_evaluation(preds, Y_test)

    return Eval, preds
