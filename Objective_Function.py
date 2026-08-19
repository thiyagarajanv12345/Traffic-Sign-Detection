import numpy as np
from Evaluation_All import seg_evaluation
from Glob_Vars import Glob_Vars
from Model_RViT_ADYv10 import Model_RViT_ADYv10


def Objective_Seg(Soln):
    Images = Glob_Vars.Images
    GT = Glob_Vars.GT
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_RViT_ADYv10(Images, GT, sol)
            Eval = seg_evaluation(pred, GT)
            Fitn[i] = 1 / Eval[5]  # IoU
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_RViT_ADYv10(Images, GT, sol)
        Eval = seg_evaluation(pred, GT)
        Fitn = 1 / Eval[5]  # IoU
        return Fitn
