import math
import time
import numpy as np
from skimage.restoration import estimate_sigma
import cv2 as cv
from scipy.stats import entropy
from skimage import filters
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = tnr / fnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(act.shape[0]):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = sum(Tp)[0]
    fp = sum(Fp)[0]
    tn = sum(Tn)[0]
    fn = sum(Fn)[0]

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (tp + fn)) * 100
    NPV = (tn / (tn + fn)) * 100
    For = (fn / (fn + tn)) * 100
    FDR = (fp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MCC = (((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) * 100
    pt = np.math.sqrt(FPR) / (np.math.sqrt(sensitivity) + np.math.sqrt(sensitivity))
    ba = (sensitivity + specificity) / 2
    fm = np.math.sqrt(sensitivity * precision)
    bm = sensitivity + specificity - 100
    mk = precision + NPV - 100
    PLHR = sensitivity / FPR
    lrminus = NegativeLivelihoodRatio(specificity, FNR)
    dor = (tp * tn) / (fp * fn)
    prevalence = ((tp + fn) / (tp + fp + tn)) * 100
    TS = (tp / (tp + fn + fp)) * 100
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC, For, pt,
            ba, fm, bm, mk, PLHR, lrminus, dor, prevalence, TS]
    # EVAL = [accuracy, precision, For, prevalence, TS, pt, bm, mk]
    return EVAL


def seg_evaluation(sp, act):
    start_time = time.time()
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn)
    Jaccard = tp / (tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    total_time = time.time() - start_time
    FPS = len(act) / total_time if total_time > 0 else 0
    MAP = (precision * sensitivity)
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    NPV = tn / (tn + fp)
    FDR = fp / (tp + fp)
    F1_score = (2 * tp) / (2 * tp + fp + fn)
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, FPS, MAP, sensitivity, specificity, precision, FPR, FNR, NPV, FDR,
            F1_score,
            MCC]
    return EVAL


def net_evaluation(sp, act):
    start_time = time.time()
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = ((2 * tp) / ((2 * tp) + fp + fn))
    Jaccard = (tp / (tp + fp + fn))
    accuracy = ((tp + tn) / (tp + tn + fp + fn))
    sensitivity = (tp / (tp + fn))
    specificity = (tn / (tn + fp))
    precision = (tp / (tp + fp))
    total_time = time.time() - start_time
    FPS = len(act) / total_time if total_time > 0 else 0
    MAP = 1 / 2 * (tp / (tp + fp))
    FPR = (fp / (fp + tn))
    FNR = (fn / (tp + fn))
    NPV = (tn / (tn + fp))
    FDR = (fp / (tp + fp))
    F1_score = ((2 * tp) / (2 * tp + fp + fn))
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, FPS, MAP, sensitivity, specificity, precision, FPR, FNR, NPV, FDR,
            F1_score,
            MCC]

    return EVAL


# Old Dehazing Metrics
# # ===================================================
# # Helper — Resize and Convert to Same Format
# # ===================================================
# def preprocess_images(original, dehazed):
#     if original.shape != dehazed.shape:
#         dehazed = cv.resize(dehazed, (original.shape[1], original.shape[0]))
#     return original, dehazed
#
#
# # ===================================================
# # 1️ Mean Squared Error (MSE)
# # ===================================================
# def MSE(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     original = original.astype(np.float64)
#     dehazed = dehazed.astype(np.float64)
#     return np.mean((original - dehazed) ** 2)
#
#
# # ===================================================
# # 2️ Root Mean Square Error (RMSE)
# # ===================================================
# def RMSE(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     return np.sqrt(MSE(original, dehazed))
#
#
# # ===================================================
# # 3️ Mean Absolute Error (MAE)
# # ===================================================
# def MAE(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     original = original.astype(np.float64)
#     dehazed = dehazed.astype(np.float64)
#     return np.mean(np.abs(original - dehazed))
#
#
# # ===================================================
# # 4️ Peak Signal-to-Noise Ratio (PSNR)
# # ===================================================
# def PSNR(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     return psnr(original, dehazed, data_range=255)
#
#
# # ===================================================
# # 5️ Structural Similarity Index Measure (SSIM)
# # ===================================================
# def SSIM(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     if len(original.shape) == 2:  # grayscale
#         return ssim(original, dehazed, data_range=255)
#     else:  # color image
#         return ssim(original, dehazed, channel_axis=-1, data_range=255)
#
#
# # ===================================================
# # 6️ Universal Image Quality Index (UIQI)
# # ===================================================
# def UIQI(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     original = original.astype(np.float64)
#     dehazed = dehazed.astype(np.float64)
#     mean_x = np.mean(original)
#     mean_y = np.mean(dehazed)
#     var_x = np.var(original)
#     var_y = np.var(dehazed)
#     cov_xy = np.mean((original - mean_x) * (dehazed - mean_y))
#     numerator = 4 * mean_x * mean_y * cov_xy
#     denominator = (mean_x ** 2 + mean_y ** 2) * (var_x + var_y)
#     return numerator / denominator if denominator != 0 else 0
#
#
# # ===================================================
# # 7️ Visual Information Fidelity (VIF)
# # ===================================================
# def VIF(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     original = original.astype(np.float64) / 255.0
#     dehazed = dehazed.astype(np.float64) / 255.0
#     sigma_nsq = 0.1
#     num, den = 0.0, 0.0
#     for scale in range(4):
#         N = 2 ** (4 - scale + 1) + 1
#         sd = N / 5.0
#         mu1 = gaussian_filter(original, sd)
#         mu2 = gaussian_filter(dehazed, sd)
#         sigma1_sq = gaussian_filter(original * original, sd) - mu1 * mu1
#         sigma2_sq = gaussian_filter(dehazed * dehazed, sd) - mu2 * mu2
#         sigma12 = gaussian_filter(original * dehazed, sd) - mu1 * mu2
#         sigma1_sq[sigma1_sq < 0] = 0
#         sigma2_sq[sigma2_sq < 0] = 0
#         g = sigma12 / (sigma1_sq + 1e-10)
#         sv_sq = sigma2_sq - g * sigma12
#         g[sigma1_sq < 1e-10] = 0
#         sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
#         sv_sq[sv_sq <= 1e-10] = 1e-10
#         num += np.sum(np.log10(1.0 + (g * g * sigma1_sq) / (sv_sq + sigma_nsq)))
#         den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))
#         # downscale for next iteration
#         if original.shape[0] > 1 and original.shape[1] > 1:
#             original = cv.resize(original, (original.shape[1] // 2, original.shape[0] // 2))
#             dehazed = cv.resize(dehazed, (dehazed.shape[1] // 2, dehazed.shape[0] // 2))
#     return num / den if den != 0 else 0
#
#
# # ===================================================
# # 8️ Feature Similarity Index Measure (FSIM)
# # ===================================================
# def FSIM(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     if len(original.shape) == 2:  # grayscale
#         gray1, gray2 = original, dehazed
#     else:
#         gray1 = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
#         gray2 = cv.cvtColor(dehazed, cv.COLOR_BGR2GRAY)
#     T1, T2 = 0.85, 160
#     PC1 = np.abs(cv.Sobel(gray1, cv.CV_64F, 1, 0, ksize=3)) + np.abs(cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=3))
#     PC2 = np.abs(cv.Sobel(gray2, cv.CV_64F, 1, 0, ksize=3)) + np.abs(cv.Sobel(gray2, cv.CV_64F, 0, 1, ksize=3))
#     S_PC = (2 * PC1 * PC2 + T1) / (PC1 ** 2 + PC2 ** 2 + T1)
#     S_I = (2 * gray1 * gray2 + T2) / (gray1 ** 2 + gray2 ** 2 + T2)
#     PC_max = np.maximum(PC1, PC2)
#     FSIM_val = np.sum(S_I * S_PC * PC_max) / np.sum(PC_max)
#     return FSIM_val
#
#
# # def FPS(original, dehazed):
# #     start_time = time.time()
# #
# #     _ = dehazing_metrics(original, dehazed)
# #
# #     end_time = time.time()
# #
# #     fps = 1.0 / (end_time - start_time)
# #     return fps
#
#
# # Combined Function for All Metrics
# def dehazing_metrics(original, dehazed):
#     original, dehazed = preprocess_images(original, dehazed)
#     Eval = [PSNR(original, dehazed), SSIM(original, dehazed),   # FPS(original, dehazed),
#             MSE(original, dehazed), RMSE(original, dehazed),
#             MAE(original, dehazed), UIQI(original, dehazed),
#             VIF(original, dehazed), FSIM(original, dehazed)]
#     return Eval


# New Dehazing Metrics

# ===================================================
# Helper — Convert to Grayscale
# ===================================================
def to_gray(image):
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


# ===================================================
# 1️⃣ BRISQUE (Simplified)
# ===================================================
def BRISQUE(dehazed):
    gray = to_gray(dehazed)
    return np.std(gray) / (np.mean(gray) + 1e-6)


# ===================================================
# 2️⃣ NIQE (Simplified)
# ===================================================
def NIQE(dehazed):
    gray = to_gray(dehazed)
    mu = np.mean(gray)
    sigma = np.std(gray)
    return sigma / (mu + 1e-6)


# ===================================================
# 3️⃣ PIQE (Simplified)
# ===================================================
def PIQE(dehazed):
    gray = to_gray(dehazed)
    var = np.var(gray)
    return var / 255.0


# ===================================================
# 4️⃣ FADE (Fog Aware Density Evaluator)
# ===================================================
def FADE(dehazed):
    gray = to_gray(dehazed)
    lap = cv.Laplacian(gray, cv.CV_64F)
    return np.mean(np.abs(lap))


# ===================================================
# 5️⃣ Entropy
# ===================================================
def Entropy(dehazed):
    gray = to_gray(dehazed)
    return shannon_entropy(gray)


# ===================================================
# 6️⃣ Contrast Gain
# ===================================================
def Contrast_Gain(dehazed):
    gray = to_gray(dehazed)
    return np.std(gray)


# ===================================================
# 7️⃣ Edge Visibility Index
# ===================================================
def Edge_Visibility_Index(dehazed):
    gray = to_gray(dehazed)

    edges = cv.Canny(gray, 100, 200)

    return np.sum(edges) / edges.size


# ===================================================
# 8️⃣ Gradient Ratio
# ===================================================
def Gradient_Ratio(dehazed):
    gray = to_gray(dehazed)

    gx = cv.Sobel(gray, cv.CV_64F, 1, 0)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1)

    grad = np.sqrt(gx ** 2 + gy ** 2)

    return np.mean(grad)


# ===================================================
# 9️⃣ Sharpness Index
# ===================================================
def Sharpness_Index(dehazed):
    gray = to_gray(dehazed)

    lap = cv.Laplacian(gray, cv.CV_64F)

    return lap.var()


# ===================================================
# 🔟 Colorfulness Metric
# ===================================================
def Colorfulness_Metric(dehazed):
    B, G, R = cv.split(dehazed)

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg = np.std(rg)
    std_yb = np.std(yb)

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    return np.sqrt(std_rg ** 2 + std_yb ** 2) + \
        0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)


# ===================================================
# 1️⃣1️⃣ Saturation Ratio
# ===================================================
def Saturation_Ratio(dehazed):
    hsv = cv.cvtColor(dehazed, cv.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]

    return np.mean(saturation) / 255.0


# ===================================================
# 1️⃣2️⃣ Haze Density Index
# ===================================================
def Haze_Density_Index(dehazed):
    gray = to_gray(dehazed)

    dark_channel = np.min(dehazed, axis=2)

    return np.mean(dark_channel) / 255.0


# ===================================================
# 1️⃣3️⃣ Structural Sharpness Measure
# ===================================================
def Structural_Sharpness(dehazed):
    gray = to_gray(dehazed)

    sobel = filters.sobel(gray)

    return np.mean(sobel)


# ===================================================
# 1️⃣4️⃣ Perceptual Fog Density
# ===================================================
def Perceptual_Fog_Density(dehazed):
    gray = to_gray(dehazed)

    sigma = estimate_sigma(gray, average_sigmas=True)

    return sigma


# ===================================================
# Combined Function
# ===================================================
def dehazing_metrics(dehazed):
    Eval = [

        BRISQUE(dehazed),
        NIQE(dehazed),
        PIQE(dehazed),
        FADE(dehazed),
        Entropy(dehazed),
        Contrast_Gain(dehazed),
        Edge_Visibility_Index(dehazed),
        Gradient_Ratio(dehazed),
        Sharpness_Index(dehazed),
        Colorfulness_Metric(dehazed),
        Saturation_Ratio(dehazed),
        Haze_Density_Index(dehazed),
        Structural_Sharpness(dehazed),
        Perceptual_Fog_Density(dehazed)

    ]

    return Eval

