from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from scipy import interpolate

def get_detection_method_string(detection_method):
    # detection method params
    minimum_duration = 100
    dispersion_threshold = 1.0
    velocity_threshold = 20.0
    
    if detection_method == 'ivt':
        detection_params = {'minimum_duration': minimum_duration,
                            'velocity_threshold': velocity_threshold,
                        }
    elif detection_method == 'idt':
        detection_params = {'minimum_duration': minimum_duration,
                            'dispersion_threshold': dispersion_threshold,
                        }
    elif detection_method == 'microsaccades':
        detection_params = {'minimum_duration': minimum_duration,
                        }
    
    detection_param_string = detection_method + '_'
    for key in detection_params:
        detection_param_string += str(key) + '_' + str(detection_params[key]) + '_'
    detection_param_string = detection_param_string[0:len(detection_param_string)-1]
    return detection_param_string

def avg_tpr_fpr_curve(fprs, tprs, label, plot_random=False,
                title = None, plot_statistics = False,
                loc = 'best', plot_legend = True,
                plot_points = 10000, ncol=1,
                bbox_to_anchor=None,
                starting_point = None,
                fontsize = 14, xscale = None,
                decimals=3,
                color = None):

    """
    Plot average roc curve from multiple fpr and tpr arrays of multiple cv-folds

    :param fprs: list of fpr arrays for the different folds
    :param tprs: list of tpr arrays for the different folds
    :label: name for the legend
    :plot_random: indicator, indicating if the random guessing curve should be plotted
    :title: title of plot; no title if 'None'
    :plot_statistics: if True, statistics for all the folds are plotted
    :loc: location of legend
    :plot_legend: if True legend is plotted
    :plot_points: number of points to plot
    :ncol: number of columns for legend
    :bbox_to_anchor: bounding box for legend outside of plot
    :starting_point: indicates the starting point of drawing the curves
    :fontsize: fontsize
    :xscale: scale for x-axis
    """
    if xscale is not None:
        plt.xscale(xscale)

    tprs_list = []
    aucs = []
    for i in range(0, len(fprs)):
        fpr = fprs[i]
        tpr = tprs[i]

        tprs_list.append(interpolate.interp1d(fpr, tpr))
        aucs.append(metrics.auc(fprs[i], tprs[i]))
    aucs = np.array(aucs)
    x = np.linspace(0, 1, plot_points)
    if starting_point is not None:
        x = x[x > starting_point]    
        
    if plot_random:
        plt.plot(x,x, color='grey', linestyle='dashed',
                 label='Random guessing')

    # plot average and std error of those roc curves:        
    ys = np.vstack([f(x) for f in tprs_list])
    ys_mean = ys.mean(axis=0)
    ys_std = ys.std(axis=0) / np.sqrt(len(fprs))
    cur_label = label
    if plot_statistics:
        cur_label += ' (AUC=' + str(np.round(np.mean(aucs), decimals)) + ' $\\pm$ ' +\
                    str(np.round(np.std(aucs) / np.sqrt(len(aucs)), decimals)) + ')'

    if color is None:
        p = plt.plot(x, ys_mean, label=cur_label)
        print(p[0].get_color())
        plt.fill_between(x, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)
    else:
        plt.plot(x, ys_mean, label=cur_label, color=color)
        plt.fill_between(x, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2, color=color)
    if plot_legend:
        if bbox_to_anchor is None:
            plt.legend(loc=loc, ncol=ncol,fontsize=fontsize)
        else:
            plt.legend(loc=loc, ncol=ncol, bbox_to_anchor = bbox_to_anchor,fontsize=fontsize)
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)

    plt.grid('on')
    if title is not None:
        plt.title(title)
        
    return aucs