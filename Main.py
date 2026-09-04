import json
import math
import os
import cv2
import numpy as np
import cv2 as cv
from Model_VDDGAN import Dehaze_Generator
from numpy import matlib
from tqdm import tqdm
import Glob_Vars
from CWO import CWO
from Flo import FLO
from IFLO import IFLO
from Model_RViT_ADYv10 import Model_RViT_ADYv10
from Model_TrafficSign import Model_TrafficSign
from Model_Yolov9 import Model_YOLOv9
from Model_Yolov8 import Model_YOLOv8
from Objective_Function import Objective_Seg
from Plot_Result import *
from TOT import TOT
from WOA import WOA

No_of_dataset = 3


def Groundtruth(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    _, threshold_image = cv.threshold(blurred_image, 120, 200, cv.THRESH_BINARY)
    return threshold_image


def Target_assign(Target):
    Tar = np.asarray(Target)
    uniq, count = np.unique(Tar, return_counts=True)
    Targets = np.zeros((len(Tar), len(uniq)))
    for i in range(len(uniq)):
        index = np.where(Tar == uniq[i])
        Targets[index, i] = 1
    return Targets


def read_Text(Text_path):
    with open(Text_path, 'r') as file:
        # Read the contents and split by new lines
        lines = file.read().splitlines()
    return lines


def pytago(width, height):
    return math.floor(math.sqrt(width * width + height * height))


# Read .json file for Target - Dataset 1
an = 0
if an == 1:
    path = './Dataset/Dataset 1/labels/det_v2_train_release.json'
    Name = []
    Cate = []
    Box = []
    # Open and read the JSON file
    with open(path, "r") as file:
        Json_data = json.load(file)
    for s, data in enumerate(Json_data):
        print(s)
        name = data.get("name", "")
        lab = data.get("labels", None)  # Get labels safely
        cat = []
        Box2d = []
        # Skip if lab is None
        if lab is None:
            continue
        for m in range(len(lab)):
            categories = lab[m].get("category", "")
            cat.append(categories)
            box2d = lab[m].get("box2d", "")
            Box2d.append(box2d)
        Name.append(name)
        Cate.append(cat)
        Box.append(Box2d)
    # Convert lists to NumPy arrays
    Name = np.array(Name, dtype=object).reshape(-1, 1)
    Cate = np.array(Cate, dtype=object).reshape(-1, 1)
    Box = np.array(Box, dtype=object).reshape(-1, 1)
    # Concatenate side by side
    result = np.column_stack((Name, Cate, Box))
    # Save as a NumPy file
    np.save('Json_target.npy', result)

# Read Dataset 1
an = 0
if an == 1:
    Target = np.load('Json_target.npy', allow_pickle=True)
    path = './Dataset/Dataset 1/BDD100K'
    Listdir = os.listdir(path)
    Images = []
    GT = []
    Targets = []
    for k in tqdm(range(len(Listdir)), desc="Dehaze Image"):
        Tar_Name = Target[k, 0]
        try:
            indices = Listdir.index(Tar_Name)
            Img_path = path + '/' + Listdir[indices]
            Image = cv.imread(Img_path)
            BBox_tar = Target[k, 2]
            output_image = np.zeros_like(Image)
            for m in range(len(BBox_tar)):
                x1 = int(BBox_tar[m]['x1'])
                x2 = int(BBox_tar[m]['x2'])
                y1 = int(BBox_tar[m]['y1'])
                y2 = int(BBox_tar[m]['y2'])
                cropped_region = Image[y1:y2, x1:x2]
                mean_values = np.mean(cropped_region, axis=(0, 1))
                cropped_image = np.clip(cropped_region * (128.0 / mean_values), 0, 255).astype(np.uint8)
                wb_cropped = Groundtruth(cropped_image)
                wb_cropped_bgr = cv.cvtColor(wb_cropped, cv.COLOR_GRAY2BGR)
                output_image[y1:y2, x1:x2] = wb_cropped_bgr
            output_rgb = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)
            Tar = Target[k, 1][0]
            Targets.append(Tar)
            Images.append(Image)
            GT.append(output_rgb)
        except:
            continue
    Tar = Target_assign(Targets)
    np.save('Images_1.npy', Images)
    np.save('Ground_Truth_1.npy', GT)
    np.save('Target_1.npy', Target)

# Read Dataset 2
an = 0
if an == 1:
    path = './Dataset/Dataset 2/images/train'
    label_path = './Dataset/Dataset 2/labels/train'
    img_dir = os.listdir(path)
    label_dir = os.listdir(label_path)
    Images = []
    GT = []
    Targets = []
    for s in range(len(img_dir)):
        print(s)
        if img_dir[s].split('.')[:-1] == label_dir[s].split('.')[:-1]:
            img_path = path + '/' + img_dir[s]
            Image = cv.imread(img_path)
            txt_path = label_path + '/' + label_dir[s]
            bbox_list = read_Text(txt_path)
            bbox = [text.split() for text in bbox_list]
            output_image = np.zeros_like(Image)
            for k in range(len(bbox)):
                label = int(bbox[k][0])
                class_id, x_center, y_center, width, height = map(float, bbox[k])
                # Convert normalized coordinates to pixel values
                image_height, image_width, _ = Image.shape
                x_center_pixel = int(x_center * image_width)
                y_center_pixel = int(y_center * image_height)
                width_pixel = int(width * image_width)
                height_pixel = int(height * image_height)

                # Calculate the top-left corner of the bounding box
                x1 = int(x_center_pixel - width_pixel / 2)
                y1 = int(y_center_pixel - height_pixel / 2)
                x2 = int(x_center_pixel + width_pixel / 2)
                y2 = int(y_center_pixel + height_pixel / 2)
                cropped_region = Image[y1:y2, x1:x2]
                mean_values = np.mean(cropped_region, axis=(0, 1))
                cropped_image = np.clip(cropped_region * (128.0 / mean_values), 0, 255).astype(np.uint8)
                wb_cropped = Groundtruth(cropped_image)
                wb_cropped_bgr = cv.cvtColor(wb_cropped, cv.COLOR_GRAY2BGR)
                output_image[y1:y2, x1:x2] = wb_cropped_bgr
                Tar = class_id
            output_rgb = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)
            Targets.append(Tar)
            Images.append(Image)
            GT.append(output_rgb)
    Tar = Target_assign(Targets)
    np.save('Images_2.npy', Images)
    np.save('Ground_Truth_2.npy', GT)
    np.save('Target_2.npy', Targets)

# Read Dataset 3
an = 0
if an == 1:
    ground_truth = {'boxes': [], 'labels': []}
    error_count = 0
    f = open('./Dataset/Dataset 3/gt.txt')
    dirname = "./Dataset/Dataset 3/TrainIJCNN2013/TrainIJCNN2013"
    ims = []
    y_true = []
    boxs = []
    labels = []
    curname = ''
    for x in f:
        datas = x.split(';')
        filename = datas[0]
        name = datas[0].split('.')[0]
        if curname != name:
            curname = name
            ground_truth['boxes'] = boxs
            ground_truth['labels'] = labels
            y_true.append(ground_truth)
            ground_truth = {'boxes': [], 'labels': []}
            boxs = []
            labels = []
            im = dirname + '/' + filename
            ims.append(im)
        width = int(datas[3]) - int(datas[1])
        height = int(datas[4]) - int(datas[2])
        diagonal = pytago(width, height)
        if diagonal >= 45.0:
            boxs.append([int(i) for i in [datas[1], datas[2], datas[3], datas[4]]])
            labels.append(int(datas[5]))
        else:
            error_count += 1
    ground_truth['boxes'] = boxs
    ground_truth['labels'] = labels
    y_true.append(ground_truth)
    y_true.pop(0)
    f.close()
    print(error_count)
    Target = []
    Images = []
    GT = []
    for n in range(len(ims)):
        image = cv2.imread(ims[n])
        Tars = y_true[n]['boxes']
        output_image = np.zeros(image.shape)
        if len(Tars) >= 1:
            Tar = Tars[0]
            x1 = int(Tar[0])
            y1 = int(Tar[1])
            x2 = int(Tar[2])
            y2 = int(Tar[3])
            lab = [x1, y1, x2, y2]
            box = (x1, y1, x2, y2)

            cropped_region = image[y1:y2, x1:x2]
            mean_values = np.mean(cropped_region, axis=(0, 1))
            cropped_image = np.clip(cropped_region * (128.0 / mean_values), 0, 255).astype(np.uint8)
            output_image[y1:y2, x1:x2] = cropped_image

            Images.append(image)
            Target.append(Tar)
            GT.append(output_image)
        else:
            Images.append(image)
            Target.append(Tars)
            GT.append(output_image)
    np.save('Images_3.npy', Images)
    np.save('Target_3.npy', Target)
    np.save('Ground_Truth_3.npy', GT)

# Image De-hazing
an = 0
if an == 1:
    for k in range(No_of_dataset):
        print(k)
        EV = []
        Images = np.load('Images_' + str(k + 1) + '.npy', allow_pickle=True)
        Preprocess = []
        Evaluate = []
        for i in range(len(Images)):
            Image = Images[i]
            Eval, prep = Dehaze_Generator(Image)
            Preprocess.append(prep)
            Evaluate.append(Eval)
        np.save('De_Hazed_Images_' + str(k + 1) + '.npy', Preprocess)
        EV.append(Evaluate)
    np.save('Eval_Seg_Act.npy', EV)

# Optimization for Detection
an = 0
if an == 1:
    Fit = []
    fitness = []
    for k in range(No_of_dataset):
        Images = np.load('De_Hazed_Images_' + str(k + 1) + '.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_' + str(k + 1) + '.npy', allow_pickle=True)
        Glob_Vars.Images = Images
        Glob_Vars.GT = GT
        fname = Objective_Seg
        Npop = 10
        Chlen = 3
        max_iter = 50
        xmin = matlib.repmat(([5, 0.01, 1]), Npop, 1)
        xmax = matlib.repmat(([255, 0.99, 5]), Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])

        print('CWO....')
        [bestfit1, fitness1, bestsol1, Time1] = CWO(initsol, fname, xmin, xmax, max_iter)

        print('TOT....')
        [bestfit2, fitness2, bestsol2, Time2] = TOT(initsol, fname, xmin, xmax, max_iter)

        print('WOA....')
        [bestfit3, fitness3, bestsol3, Time3] = WOA(initsol, fname, xmin, xmax, max_iter)

        print('FLO....')
        [bestfit4, fitness4, bestsol4, Time4] = FLO(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = IFLO(initsol, fname, xmin, xmax, max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(k + 1) + '.npy', BestSol)

        BestFit = [bestfit1, bestfit2, bestfit3, bestfit4, bestfit5]
        np.save('BestFit_' + str(k + 1) + '.npy', BestFit)

        Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
        Fit.append(Fitness)
        Time = [Time1, Time2, Time3, Time4, Time5]
        np.save('Time_' + str(k + 1) + '.npy', Time)

    np.save('Fitness.npy', fitness)

# Detection
an = 0
if an == 1:
    Eval = []
    for s in range(No_of_dataset):
        Images = np.load('De_Hazed_Images_' + str(s + 1) + '.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_' + str(s + 1) + '.npy', allow_pickle=True)
        sol = np.load('BestSol_' + str(s + 1) + '.npy', allow_pickle=True)  # [s]
        Epoch = [20, 30, 40, 50, 60]
        for m in range(len(Epoch)):
            EVAL = np.zeros((10, 16))
            for i in range(5):  # for all algorithms
                EVAL[i, :], pred = Model_RViT_ADYv10(Images, GT, Epoch[m], sol[i].astype('int'))
            EVAL[5, :] = Model_TrafficSign(Images, GT, Epoch[m])
            EVAL[6, :] = Model_YOLOv8(Images, GT, Epoch[m])
            EVAL[7, :] = Model_YOLOv9(Images, GT, Epoch[m])
            EVAL[8, :], pred8 = Model_RViT_ADYv10(Images, GT, Epoch[m])
            EVAL[9, :] = EVAL[4, :]
            Eval.append(EVAL)
            np.save('Detected_Images_' + str(s + 1) + '.npy', pred)
    np.save('Eval_all.npy', Eval)

plotConvResults()
Plots_Results()
Plot_Proposed_Results()
plot_seg_results()
Table()
Seg_Table()
