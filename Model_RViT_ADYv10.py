import numpy as np
import cv2 as cv
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Conv2D, MaxPooling2D, Add, Multiply,
                          UpSampling2D, Concatenate, Dense, Dropout,
                          LayerNormalization, Lambda)
from keras.optimizers import Adam
from Evaluation_All import seg_evaluation


# Helper: simple MLP used inside small transformer blocks
def mlp_block(x, hidden_units, dropout=0.0):
    x = Dense(hidden_units, activation='gelu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(hidden_units, activation='gelu')(x)
    return x


# Regional ViT (patch / local attention) block with dilation option
# - inputs: (B, H, W, C) with known H,W at build time
# - region_size must divide H and W exactly
def Regional_ViT_Dilated(inputs, region_size=8, num_heads=4, head_dim=32, mlp_dim=128, dilation_rates=(1, 2)):
    H, W, C = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    grid_h = H // region_size
    grid_w = W // region_size
    tokens = region_size * region_size

    # Dilated convs
    x = Conv2D(C, 3, padding='same', dilation_rate=dilation_rates[0], activation='relu')(inputs)
    x = Conv2D(C, 3, padding='same', dilation_rate=dilation_rates[1], activation='relu')(x)

    # Reshape to regions (B, grid_h, region_h, grid_w, region_w, C)
    x_regions = tf.keras.layers.Reshape((grid_h, region_size, grid_w, region_size, C))(x)
    # Merge region dims (B, grid_h, grid_w, tokens, C)
    x_regions = tf.keras.layers.Reshape((grid_h, grid_w, tokens, C))(x_regions)
    # Flatten grids to batch dimension (B*grid_h*grid_w, tokens, C)
    x_regions = tf.keras.layers.Reshape((-1, tokens, C))(x_regions)

    # Attention
    embed_dim = num_heads * head_dim
    token_emb = Dense(embed_dim)(x_regions)
    att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(token_emb, token_emb)
    att = LayerNormalization()(att + token_emb)
    mlp = Dense(mlp_dim, activation='gelu')(att)
    out_tokens = LayerNormalization()(att + mlp)

    # Project back
    region_proj = Dense(tokens * C)(out_tokens)
    # region_proj = tf.keras.layers.Reshape((grid_h, grid_w, region_size, region_size, C))(regio
    grid_h = 8
    grid_w = 8
    region_size = 8
    C = 8192

    region_proj = tf.keras.layers.Reshape((grid_h, grid_w, region_size, region_size, C))(region_proj)
    region_proj = tf.keras.layers.Reshape((H, W, C))(region_proj)

    # fused = Add()([inputs, region_proj])
    region_proj = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')(region_proj)
    fused = tf.keras.layers.Add()([inputs, region_proj])
    return fused


# Multiscale Attentional Feature Fusion (MAF-Fusion) layer
def MAF_Fusion(inputs):
    # low, medium, high channels
    low_scale = Conv2D(256, 3, activation='relu', padding='same')(inputs)
    medium_scale = Conv2D(256, 3, activation='relu', padding='same')(inputs)
    high_scale = Conv2D(256, 3, activation='relu', padding='same')(inputs)

    low_attention = Conv2D(1, 1, activation='sigmoid', padding='same')(low_scale)
    medium_attention = Conv2D(1, 1, activation='sigmoid', padding='same')(medium_scale)
    high_attention = Conv2D(1, 1, activation='sigmoid', padding='same')(high_scale)

    low_scale_fused = Multiply()([low_scale, low_attention])
    medium_scale_fused = Multiply()([medium_scale, medium_attention])
    high_scale_fused = Multiply()([high_scale, high_attention])

    def Fusion_Block(low, med, high):
        # Target size is based on the highest-resolution input
        target_h = tf.shape(high)[1]
        target_w = tf.shape(high)[2]

        # Dynamically resize lower-resolution feature maps to match 'high'
        low_resized = tf.image.resize(low, (target_h, target_w), method='bilinear')
        med_resized = tf.image.resize(med, (target_h, target_w), method='bilinear')
        high_resized = high

        # Concatenate along channel axis
        fusion = Concatenate()([low_resized, med_resized, high_resized])  # (None, target_h, target_w, combined_channels)

        # Reduce to 3-channel output
        fusion = Conv2D(256, 1, activation='relu', padding='same')(fusion)
        fusion = Conv2D(128, 3, activation='relu', padding='same')(fusion)
        fusion = Conv2D(64, 3, activation='relu', padding='same')(fusion)
        fusion_output = Conv2D(3, 1, activation='sigmoid', padding='same')(fusion)
        return fusion_output

    fusion_output = Fusion_Block(low_scale_fused, medium_scale_fused, high_scale_fused)
    return fusion_output


def yolo_v10_regional_vit_seg(sol, input_shape=(512, 512, 3), num_classes=3, region_size=8):
    inputs = Input(input_shape)

    # Backbone / encoder (kept in your coding style)
    x = Conv2D(32, 3, activation=int(sol[2]), padding='same')(inputs)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    p1 = MaxPooling2D(pool_size=(2, 2))(x)  # 256x256

    x = Conv2D(64, 3, activation='relu', padding='same')(p1)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    p2 = MaxPooling2D(pool_size=(2, 2))(x)  # 128x128

    x = Conv2D(128, 3, activation='relu', padding='same')(p2)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    p3 = MaxPooling2D(pool_size=(2, 2))(x)  # 64x64

    # Apply Regional ViT with dilation on mid-level features (64x64)
    rvit = Regional_ViT_Dilated(p3, region_size=region_size, num_heads=4, head_dim=32, mlp_dim=128, dilation_rates=(1, 2))

    # deeper convs and further downsampling
    x = Conv2D(int(sol[0]), 3, activation='relu', padding='same')(rvit)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    p4 = MaxPooling2D(pool_size=(2, 2))(x)  # 32x32

    # bottleneck
    b = Conv2D(512, 3, activation='relu', padding='same')(p4)
    b = Conv2D(512, 3, activation='relu', padding='same')(b)

    # Decoder (upsample)
    u1 = UpSampling2D((2, 2), interpolation='bilinear')(b)  # 64x64
    u1 = Concatenate()([u1, x])
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(u1)  # 128x128
    u2 = Concatenate()([u2, p2])
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(u2)  # 256x256
    u3 = Concatenate()([u3, p1])
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)

    u4 = UpSampling2D((2, 2), interpolation='bilinear')(u3)  # 512x512
    u4 = Concatenate()([u4, inputs])
    u4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    u4 = Conv2D(64, 3, activation='relu', padding='same')(u4)

    # MAF fusion and final projection
    fused = MAF_Fusion(u4)

    # final output: per-pixel mask with num_classes channels (3 for RGB masks)
    out = Conv2D(num_classes, 1, activation='sigmoid', padding='same')(fused)

    model = Model(inputs=inputs, outputs=out, name='RViT_Dilated_YOLOv10_Seg')
    return model


# Main training
def Model_RViT_ADYv10(Images, Groundtruth, Epoch, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]


    N = Images.shape[0]

    IMG_SIZE = 512

    # Prepare X and Y arrays
    X = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    Y = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    for i in range(N):
        im = Images[i]
        gt = Groundtruth[i]
        imr = cv.resize(im, (IMG_SIZE, IMG_SIZE))
        gtr = cv.resize(gt, (IMG_SIZE, IMG_SIZE))

        # normalize to [0,1]
        X[i] = imr.astype(np.float32) / 255.0
        Y[i] = gtr.astype(np.float32) / 255.0

    # Build model
    num_classes = Y.shape[-1]  # expected 3
    model = yolo_v10_regional_vit_seg(sol, input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, region_size=8)
    model.summary()

    # Loss & compile
    # If your groundtruth is RGB mask (independent channels), BCE works fine. Otherwise change accordingly.
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=Adam(learning_rate=int(sol[1])), loss=loss, metrics=['accuracy'])

    # Callbacks
    earlystopper = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
    checkpointer = ModelCheckpoint('rvit_adyv10_best.keras', save_best_only=True, verbose=1)

    # Fit
    model.fit(X, Y, validation_split=0.1, batch_size=10, epochs=Epoch,
              callbacks=[earlystopper, checkpointer])

    # Predict on holdout (last 10%) for evaluation
    split_idx = int(0.9 * N)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    preds = model.predict(X_test, batch_size=10)
    Eval = seg_evaluation(preds, Y_test)

    return Eval, preds
