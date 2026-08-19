import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from matplotlib import pylab

No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CWO-RViT-ADYv10', 'TOT-RViT-ADYv10', 'WOA-RViT-ADYv10', 'FLO-RViT-ADYv10',
                 'PSFLO-RViT-ADYv10']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.legend(loc=1, fontsize=12)
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def plot_seg_results():
    eval = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'FPS', 'MAP', 'Sensitivity', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Classifier = ['Yolov8', 'YOLOv9', 'YOLOX-Tiny-RCNN', 'Yolov10', 'PSFLO-RViT-ADYv10']
    Algorithm = ['CWO-RViT-ADYv10', 'TOT-RViT-ADYv10', 'WOA-RViT-ADYv10', 'FLO-RViT-ADYv10', 'PSFLO-RViT-ADYv10']
    Labels = np.asarray(['Ref 6', 'Proposed'])
    Epoch = ['Linear', 'ReLU', 'Softmax', 'Sigmoid', 'Thnh']
    Epoch = np.asarray(Epoch)
    Graph_Terms = [0, 1, 2, 3, 4]
    colors = ['r', 'm', 'y', 'brown', 'k']

    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, 0, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Algorithm Comparison of Kfold')
            X = np.arange(len(Epoch) - 2)
            plt.barh(X + 0.00, Graph[:3, 0], color='r', height=0.15, label=Algorithm[0])
            plt.barh(X + 0.15, Graph[:3, 1], color='m', height=0.15, label=Algorithm[1])
            plt.barh(X + 0.30, Graph[:3, 2], color='y', height=0.15, label=Algorithm[1])
            plt.barh(X + 0.45, Graph[:3, 3], color='brown', height=0.15, label=Algorithm[1])
            plt.barh(X + 0.60, Graph[:3, 4], color='k', height=0.15, label=Algorithm[1])
            plt.yticks(X + 0.15, ['Linear', 'ReLU', 'Softmax'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.ylabel('Activation Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)

            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                           in colors]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                       frameon=False, ncol=len(Algorithm), prop={'weight': 'bold', 'size': 9})
            plt.tight_layout()
            plt.grid(axis='x')
            path = "./Results/Detection_Dataset_%s_%s_Seg_Alg_bar.png" % (n + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, 0, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Kfold')
            X = np.arange(len(Epoch) - 2)
            plt.barh(X + 0.00, Graph[:3, 5], color='r', height=0.15, label=Classifier[0])
            plt.barh(X + 0.15, Graph[:3, 6], color='m', height=0.15, label=Classifier[1])
            plt.barh(X + 0.30, Graph[:3, 7], color='y', height=0.15, label=Classifier[1])
            plt.barh(X + 0.45, Graph[:3, 8], color='brown', height=0.15, label=Classifier[1])
            plt.barh(X + 0.60, Graph[:3, 4], color='k', height=0.15, label=Classifier[1])
            plt.yticks(X + 0.15, ['Linear', 'ReLU', 'Softmax'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.ylabel('Activation Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)

            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
                       frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            plt.grid(axis='x')
            path = "./Results/Detection_Dataset_%s_%s_Seg_mod_bar.png" % (n + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plots_Results():
    eval = np.load('Eval_dehazing.npy', allow_pickle=True)
    Terms = ['BRISQUE', 'NIQE', 'PIQE', 'FADE', 'Entropy', 'Contrast gain', 'Edge Visibility Index', 'Gradient Ratio',
             'Sharpness Index', 'Colorfulness Metric', 'Saturation Ratio', 'Haze Density Index',
             'Structural Sharpness Measure', 'Perceptual Fog Density']

    Graph_Terms = [0, 1, 2, 3, 4, 8, 9, 12]
    bar_width = 0.15
    Optimizer = ['Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Sgd']
    # Algorithm = ['CWO', 'TOT', 'WOA', 'FLO', 'IFLO']
    Classifier = ['SRGAN', 'GAN', 'Cycle GAN', 'Conditional GAN', 'VDDGAN']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[:, l] = eval[i, :, l, Graph_Terms[j]]

            # fig = plt.figure(figsize=(12, 6))
            # ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            # fig.canvas.manager.set_window_title('Algorithm Comparison of KFold')
            # X = np.arange(len(Optimizer))
            # bars1 = plt.bar(X + 0.00, Graph[:5, 0], color='darkblue', edgecolor='w', linewidth=2, width=0.15,
            #                 label=Algorithm[0])
            # bars2 = plt.bar(X + 0.15, Graph[:5, 1], color='#9400d3', edgecolor='w', linewidth=2, width=0.15,
            #                 label=Algorithm[1])
            # bars3 = plt.bar(X + 0.30, Graph[:5, 2], color='#a30046', edgecolor='w', linewidth=2, width=0.15,
            #                 label=Algorithm[2])
            # bars4 = plt.bar(X + 0.45, Graph[:5, 3], color='#00bbf9', edgecolor='w', linewidth=2, width=0.15,
            #                 label=Algorithm[3])
            # bars5 = plt.bar(X + 0.60, Graph[:5, 4], color='k', edgecolor='w', linewidth=2, width=0.15,
            #                 label=Algorithm[4])

            # plt.xticks(X + bar_width * 2, ['Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Sgd'], fontsize=15,
            #            fontname="Arial",
            #            fontweight='bold', color='k')
            # plt.xlabel('Optimizer', fontname="Arial", fontsize=18, fontweight='bold', color='k')
            # plt.ylabel(Terms[Graph_Terms[j]], fontsize=18, fontname="Arial", fontweight='bold', color='k')
            # plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#35530a')
            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['left'].set_visible(False)
            # plt.gca().spines['bottom'].set_visible(False)
            # dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
            #                in ['darkblue', '#9400d3', '#a30046', '#00bbf9', 'k']]
            # plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
            #            frameon=False, ncol=len(Algorithm))
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            # plt.tight_layout()
            # path = "./Results/Dataset_%s_%s_Alg_bars.png" % (i + 1, Terms[Graph_Terms[j]])
            # plt.savefig(path)
            # plt.show()

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Epochs')
            X = np.arange(len(Optimizer))
            bars1 = plt.bar(X + 0.00, Graph[:5, 5], color='r', edgecolor='w', width=0.15, label=Classifier[0])
            bars2 = plt.bar(X + 0.15, Graph[:5, 6], color='b', edgecolor='w', width=0.15, label=Classifier[1])
            bars3 = plt.bar(X + 0.30, Graph[:5, 7], color='y', edgecolor='w', width=0.15,
                            label=Classifier[2])
            bars4 = plt.bar(X + 0.45, Graph[:5, 8], color='m', edgecolor='w', width=0.15, label=Classifier[3])
            bars5 = plt.bar(X + 0.60, Graph[:5, 4], color='k', edgecolor='w', width=0.15, label=Classifier[4])
            for bars in [bars1, bars2, bars3, bars4, bars5]:
                for bar in bars:
                    height = bar.get_height()

            plt.xticks(X + bar_width * 2, ['Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Sgd'], fontname="Arial",
                       fontsize=18,
                       fontweight='bold', color='k')
            plt.xlabel('Optimizer', fontname="Arial", fontsize=18, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=18, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=18, fontweight='bold', color='k')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['r', 'b', 'y', 'm', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
                       frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 14})
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/Dehazing_Dataset_%s_%s_mod_bars.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Proposed_Results():
    eval = np.load('Eval_dehazing.npy', allow_pickle=True)
    Terms = ['BRISQUE', 'NIQE', 'PIQE', 'FADE', 'Entropy', 'Contrast gain', 'Edge Visibility Index', 'Gradient Ratio',
             'Sharpness Index', 'Colorfulness Metric', 'Saturation Ratio', 'Haze Density Index',
             'Structural Sharpness Measure', 'Perceptual Fog Density']
    Kfold = ['Image1', 'Image2', 'Image3', 'Image4', 'Image5']
    Graph_Terms = [0, 1, 2, 3, 4, 8, 9, 12]

    bar_colors = ['#003f5c', '#bc5090', '#ffa600', '#58508d', '#2ca02c']
    methods = ['SRGAN', 'GAN', 'Cycle GAN', 'Conditional GAN', 'VDDGAN']

    eval_algorithms = eval[:, :, :5, :]
    eval_methods = eval[:, :, 5:, :]

    def plot_results(eval_data, labels, bar_colors, plot_type):
        for n in range(eval_data.shape[0]):  # loop over datasets
            for j in range(len(Graph_Terms)):  # loop over selected metrics

                Graph = eval_data[n, :5, :, Graph_Terms[j]]  # shape (5 folds × models)
                values = Graph.T  # shape (models × 5 folds)

                x = np.arange(len(Kfold))  # positions for folds
                bar_width = 0.18  # bar size

                fig, ax = plt.subplots(figsize=(12, 7))

                for i in range(len(labels)):
                    bars = ax.bar(x + i * bar_width, values[i], width=bar_width,
                                  color=bar_colors[i], label=labels[i])

                    # Add value labels inside bars
                    for bar in bars:
                        y = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, y / 2,
                                f"{y:.2f}",
                                ha='center', va='center',
                                fontsize=10, fontweight='bold', color='black',
                                bbox=dict(facecolor='white', edgecolor='black',
                                          boxstyle='round,pad=0.2'))

                ax.set_ylabel(Terms[Graph_Terms[j]] + ' →',
                              fontsize=15, fontweight='bold', color='#35530a')
                ax.set_xticks(x + (len(labels) - 1) * bar_width / 2)
                ax.set_xticklabels(Kfold, fontname="Arial", fontsize=18, fontweight='bold', color='k')

                ax.set_axisbelow(True)
                ax.yaxis.grid(True, linestyle='--', alpha=0.4)
                plt.yticks(fontname="Arial", fontsize=18, fontweight='bold', color='k')

                dot_markers = [plt.Line2D([0], [0], marker='s', color='w',
                                          markerfacecolor=c, markersize=10) for c in bar_colors]
                ax.legend(dot_markers, labels,
                          loc='upper center', bbox_to_anchor=(0.5, 1.10),
                          fontsize=12, frameon=False, ncol=len(labels))

                plt.tight_layout()
                path = f"./Results/Dehazing_{plot_type}_Dataset{n + 1}_{Terms[Graph_Terms[j]]}_GroupedBars.png"
                fig = pylab.gcf()
                fig.canvas.manager.set_window_title(
                    f'{plot_type} - Dataset-{n + 1} Kfold vs {Terms[Graph_Terms[j]]}'
                )
                plt.savefig(path, bbox_inches="tight")
                plt.show()

    plot_results(eval_methods, methods, bar_colors, plot_type="Methods")


def Table():
    eval = np.load('Eval_dehazing.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'CWO-RViT-ADYv10', 'TOT-RViT-ADYv10', 'WOA-RViT-ADYv10', 'FLO-RViT-ADYv10',
                 'PSFLO-RViT-ADYv10']
    Classifier = ['BatchSize', 'SRGAN', 'GAN', 'Cycle GAN', 'Conditional GAN', 'VDDGAN']
    Terms = ['BRISQUE', 'NIQE', 'PIQE', 'FADE', 'Entropy', 'Contrast gain', 'Edge Visibility Index', 'Gradient Ratio',
             'Sharpness Index', 'Colorfulness Metric', 'Saturation Ratio', 'Haze Density Index',
             'Structural Sharpness Measure', 'Perceptual Fog Density']
    Graph_Terms = np.array([0, 1, 2, 3, 4, 8, 9, 12]).astype(int)
    Table_Terms = [0, 1, 2, 3, 4, 8, 9, 12]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['4', '8', '16', '32', '48']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, :]

            # Table = PrettyTable()
            # Table.add_column(Algorithm[0], Batchsize)
            # for j in range(len(Algorithm) - 1):
            #     Table.add_column(Algorithm[j + 1], value[:5, j, Graph_Terms[k]])
            # print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison for Dehazing',
            #       '---------------------------------------')
            # print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:5, len(Classifier) + j - 1, Graph_Terms[k]])
            print('---------------------------------------Dataset - ', i + 1, table_terms[k],
                  '  Comparison for Dehazing',
                  '---------------------------------------')
            print(Table)


def Seg_Table():
    eval = np.load('Eval_all_Seg_Table.npy', allow_pickle=True)

    Algorithm = [
        'BatchSize',
        'CWO-RViT-ADYv10',
        'TOT-RViT-ADYv10',
        'WOA-RViT-ADYv10',
        'FLO-RViT-ADYv10',
        'PSFLO-RViT-ADYv10'
    ]

    Classifier = [
        'BatchSize',
        'DCNN',
        'YoloV8',
        'YoloV9',
        'RViT-ADYv10',
        'RViT-ADYv10'
    ]

    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'FPS', 'MAP', 'Sensitivity', 'Specificity', 'Precision', 'FPR',
     'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']

    # Metrics used for tables
    Graph_Terms = np.array([0,1,2]).astype(int)

    # Only first 3 metrics for table
    Table_Terms = [0,1,2]

    table_terms = [Terms[i] for i in Table_Terms]

    Batchsize = ['4', '8', '16', '32', '48']

    for i in range(eval.shape[0]):

        # Extract once per dataset
        value = eval[i, :, :, 0, 4:]
        temp = value[:, 4, :]
        value[:, 9, :] = temp
        for k in range(len(Table_Terms)):

            metric_index = Graph_Terms[k]

            # Algorithm Table

            Table = PrettyTable()

            Table.add_column(
                Algorithm[0],
                Batchsize
            )

            for j in range(len(Algorithm) - 1):
                Table.add_column(
                    Algorithm[j + 1],
                    value[:5, j, metric_index]
                )

            print(
                '------------------------------------- Dataset -',
                i + 1,
                table_terms[k],
                'Algorithm Comparison for Detected',
                '-------------------------------------'
            )

            print(Table)

            # Classifier Table

            Table = PrettyTable()

            Table.add_column(
                Classifier[0],
                Batchsize
            )

            for j in range(len(Classifier) - 1):
                Table.add_column(
                    Classifier[j + 1],
                    value[:5, len(Algorithm) + j - 1, metric_index]
                )

            print('------------------------------------- Dataset -', i + 1, table_terms[k],
                  'Classifier Comparison for Detected', '-------------------------------------')

            print(Table)


if __name__ == '__main__':
    # plotConvResults()  # Detected
    # Plots_Results()  # DeHazing
    # Plot_Proposed_Results()  # DeHazing
    # Table()  # DeHazing
    # plot_seg_results() # Detected
    Seg_Table()  # Detected
