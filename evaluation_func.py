import numpy as np
import seaborn as sns
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import ipdb
import os
from sklearn.metrics import roc_curve
from train_hd_ssl import windowing


curr_dir = os.getcwd()
VIZUALIZE_DIR = os.path.join(curr_dir, 'model_outputs', 'results_visualization')
os.makedirs(VIZUALIZE_DIR, exist_ok=True)

def confusion_matrix(labels, predictions, prefix1='', prefix2='', output_dir=None):
    if len(labels) == 0:
        return
    cm = metrics.confusion_matrix(labels, predictions) 
    try:
        recall = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-10)
        precision = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-10) if (cm[1,1] + cm[0,1]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except:
        # ipdb.set_trace()
        recall = 0
        precision = 0 
        f1_score = 0
    class_labels = ["Non-Walking", "Walking"]
    #plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={"fontsize": 16,"ha": "center", "va": "center"})
    ax.set_xticklabels(class_labels, fontsize=16, ha="center")
    ax.set_yticklabels(class_labels, fontsize=16, va="center") 
    
    plt.xlabel("Predicted",{"fontsize":16})
    plt.ylabel("True",{"fontsize":16})
    plt.title(f"Confusion Matrix Recall:{recall:.2f},Precision:{precision:.2f},F1:{f1_score:.2f}")
    output_dir = VIZUALIZE_DIR if output_dir is None else output_dir
    plt.savefig(os.path.join(output_dir,f'confusion_matrix_{prefix1}_{prefix2}.png'))
    plt.close('all')
    return recall

def generate_confusion_matrix_per_chorea_lvl(gait_predictions, gait_labels, chorea_predictions, chorea_labels, valid_chorea, valid_gait, fold_index, model_type, analysis_type='per_pixel', prefix=''):

    if model_type == 'segmentation':
        if analysis_type == 'per_window':
            gait_predictions, gait_labels_ind, chorea_labels_ind, valid_chorea = windowing(gait_predictions, gait_labels, chorea_labels, valid_chorea, valid_gait)
        if analysis_type == 'per_pixel':
            valid_gait_f = valid_gait.flatten()
            valid_gait_ind = np.where(valid_gait_f)[0]
            gait_predictions = gait_predictions.flatten()[valid_gait_ind]
            gait_labels_ind = gait_labels.flatten()[valid_gait_ind]
            chorea_labels_ind = chorea_labels.flatten()[valid_gait_ind]
            valid_chorea = valid_chorea.flatten()[valid_gait_ind]
    else:
        gait_labels_ind = torch.argmax(gait_labels, dim=-1)
        chorea_labels_ind = torch.argmax(chorea_labels, dim=-1)
    recall_dict = {}
    for is_valid in [0, 1]:
        valid_ind = np.where(valid_chorea == is_valid)[0]
        if is_valid:
            for chorea_level in np.unique(chorea_labels_ind):
                indices = np.where((chorea_labels_ind==chorea_level).flatten() * (valid_chorea == is_valid).flatten())[0]
                gait_predictions_sel = gait_predictions[indices]
                gait_labels_sel = gait_labels_ind[indices]
                if len(gait_labels_sel) > 0:
                    prefix1 = f'{fold_index}'
                    if analysis_type == 'per_pixel':
                        prefix2=f'{chorea_level}_per_pixel'
                    else:
                        prefix2=f'{chorea_level}'
                    recall = confusion_matrix(gait_labels_sel, gait_predictions_sel, prefix1=prefix1, prefix2=prefix2+prefix)
                    recall_dict[prefix2+prefix] = recall
        else:
            gait_predictions_sel = gait_predictions[valid_ind]
            gait_labels_sel = gait_labels_ind[valid_ind]
            prefix1 = f'{fold_index}'
            if analysis_type == 'per_pixel':
                prefix2=f'no_valid_chorea_per_pixel'
            else:
                prefix2=f'no_valid_chorea'
            recall_dict[prefix2+prefix] = confusion_matrix(gait_labels_sel, gait_predictions_sel, prefix1=prefix1, prefix2=f'no_valid_chorea'+prefix)
    return recall_dict

def auc_and_ci(labels,probs):

## calculation of AUC and ci as explined in the paper
    
    # labels = labels.flatten()
    # prediction_exp = np.exp(predictions)
    # prob = prediction_exp[:,1,:] / np.sum(prediction_exp, axis=1)
    # prob = prob.flatten()
    # valid = valid.flatten()
    # valid_labels = labels[valid==1]
    # valid_prob = prob[valid==1]
    try:
        Y = probs[labels==1]
        X = probs[labels==0]
        n_x = len(X)
        n_y = len(Y)
        Y_exp = np.expand_dims(Y,axis=0)
        X_exp = np.expand_dims(X,axis=1)
        ge = Y_exp > X_exp
        eq = Y_exp == X_exp
        theta = (np.sum(ge) + 0.5*np.sum(eq))/(n_x * n_y + 1e-10)
        print(f"theta:{theta}")
        q1 = np.sum((np.sum(ge, axis=1) + 0.5*np.sum(eq, axis=1))**2)/(n_x*n_y**2 + 1e-10)
        q2 = np.sum((np.sum(ge, axis=0) + 0.5*np.sum(eq, axis=0))**2)/(n_y*n_x**2 + 1e-10)
        print(f"Q1:{q1}, Q2:{q2}")
        equal_prob = np.sum(eq)/(n_x * n_y + 1e-10)
        var = 1/(n_y-1)/(n_x-1)*(theta*(1-theta) - 1/4*equal_prob + (n_y-1)*(q1-theta**2) + (n_x-1)*(q2-theta**2))
        z=1.96
        ci = z*np.sqrt(var)
        print(f"var:{var}")
    except:
        theta = None
        ci = None
    
    # compare to auc sklearn
    try:
        sklearn_auc = metrics.roc_auc_score(labels, probs)
        print(f"sklearn_auc:{sklearn_auc}")
        precision, recall, _ = metrics.precision_recall_curve(labels, probs)
        pr_auc = metrics.auc(recall, precision)
        print(f"AUC of Precision-Recall Curve: {pr_auc}")
    except:
        pass
    return theta, ci

def auc_and_ci_per_chorea_lvl(gait_logits, gait_labels, chorea_labels, valid_chorea, valid_gait, model_type, analysis_type='per_pixel'):
    gait_logits_exp = np.exp(gait_logits)
    if model_type == 'segmentation':
        gait_prob = gait_logits_exp[:,1,:] / np.sum(gait_logits_exp, axis=1)
        if analysis_type == 'per_window':
            gait_prob, gait_labels_ind, chorea_labels_ind, valid_chorea = windowing(gait_prob, gait_labels, chorea_labels, valid_chorea, valid_gait)
        if analysis_type == 'per_pixel':
            valid_gait_f = valid_gait.flatten()
            valid_gait_ind = np.where(valid_gait_f)[0]
            gait_prob = gait_prob.flatten()[valid_gait_ind]
            gait_labels_ind = gait_labels.flatten()[valid_gait_ind]
            chorea_labels_ind = chorea_labels.flatten()[valid_gait_ind]
            valid_chorea = valid_chorea.flatten()[valid_gait_ind]
    else:
        gait_prob = gait_logits_exp[:,1] / np.sum(gait_logits_exp, axis=1)
        gait_labels_ind = torch.argmax(torch.Tensor(gait_labels), dim=-1)
        chorea_labels_ind = torch.argmax(torch.Tensor(chorea_labels), dim=-1)
    auc_dict = {}
    for is_valid in [0, 1]:
        valid_ind = np.where(valid_chorea == is_valid)[0]
        if is_valid:
            for chorea_level in np.unique(chorea_labels_ind):
                indices = np.where((chorea_labels_ind==chorea_level).flatten() * (valid_chorea == is_valid).flatten())[0]
                gait_prob_sel = gait_prob[indices]
                gait_labels_sel = gait_labels_ind[indices]
                if len(gait_labels_sel) > 0:
                    if analysis_type == 'per_pixel':
                        prefix2=f'{chorea_level}_per_pixel'
                    else:
                        prefix2=f'{chorea_level}'
                    _update_dict_res(auc_dict, prefix2, gait_labels_sel, gait_prob_sel)
            all_prefix = 'all_chorea'
            all_prefix = all_prefix + ('_per_pixel' if analysis_type == 'per_pixel' else '')
            _update_dict_res(auc_dict, all_prefix, gait_labels_ind, gait_prob)
            auc, ci = auc_and_ci(gait_labels_sel,gait_prob_sel) #change to logits
            cm = metrics.confusion_matrix(gait_labels_sel,gait_prob_sel>0.5)
            auc_dict[prefix2] = {'auc':auc,
                                 'ci':ci}
        else:
            gait_prob_sel = gait_prob[valid_ind]
            gait_labels_sel = gait_labels_ind[valid_ind]
            prefix1 = f''
            if analysis_type == 'per_pixel':
                prefix2=f'no_valid_chorea_per_pixel'
            else:
                prefix2=f'no_valid_chorea'
                _update_dict_res(auc_dict, prefix2, gait_labels_sel, gait_prob_sel)
                auc, ci = auc_and_ci(gait_labels_sel,gait_prob_sel) #change to logits
                auc_dict[prefix2] = {'auc':auc,
                                     'ci':ci}   
    return auc_dict

def _update_dict_res(res_dict, key, labels, prob):
    print(key)
    auc, ci = auc_and_ci(labels,prob) #change to logits
    sklearn_auc = metrics.roc_auc_score(labels, prob)
    precision, recall, _ = metrics.precision_recall_curve(labels, prob)
    pr_auc = metrics.auc(recall, precision)
    #plot_curves(labels, prob)
    cm = metrics.confusion_matrix(labels,prob>0.5)
    recall = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-10)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    res_dict[key] = {'auc':auc,
                     'ci':ci,
                     'recall':recall,
                     'precision': precision,
                     'f1_score':f1_score,
                     'sklearn-roc-auc': sklearn_auc,
                     'PR-RC auc':pr_auc
                     }

    def update_scores_file(scores_file, new_config_name, new_scores):
        try:
            # Load existing scores from file
            with open(scores_file, 'r') as f:
                existing_scores = json.load(f)
        except FileNotFoundError:
            existing_scores = {}

        # Update existing scores with new configuration
        existing_scores[new_config_name] = new_scores

        # Save updated scores back to the file
        with open(scores_file, 'w') as f:
            json.dump(existing_scores, f, indent=4)

def find_optimal_threshold_roc(pred_probs, true_labels,subject_name,plot_roc=False,output_dir=None):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    
    # Calculate Youden's J statistic
    youden_j = tpr - fpr
    optimal_threshold_index = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_threshold_index]
    roc_auc = metrics.auc(fpr, tpr)
    if plot_roc:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', marker='o', label=f'Optimal threshold = {optimal_threshold:.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic_{subject_name}_optimal_thr')
        plt.legend(loc="lower right")
        output_dir = VIZUALIZE_DIR if output_dir is None else output_dir
        plt.savefig(os.path.join(output_dir,f'Receiver Operating Characteristic_{subject_name}.png'))
    return optimal_threshold

def plot_curves(labels, probs):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    precision, recall, _ = metrics.precision_recall_curve(labels, probs)
    np.save(os.path.join(VIZUALIZE_DIR,f'fpr.npy'),fpr)
    np.save(os.path.join(VIZUALIZE_DIR,f'tpr.npy'),tpr)
    np.save(os.path.join(VIZUALIZE_DIR,f'precision.npy'),precision)
    np.save(os.path.join(VIZUALIZE_DIR,f'recall.npy'),recall)

    ipdb.set_trace()
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save the ROC curve
    plt.savefig(os.path.join(VIZUALIZE_DIR,'roc_curve.png'))

    # Optionally, show the plot if you also want to visualize it
    plt.show()

    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % metrics.auc(recall, precision))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Save the Precision-Recall curve
    plt.savefig(os.path.join(VIZUALIZE_DIR,'pr_curve.png'))

def illustrate_results(cv_test_idxs_all_folds, win_subjects, win_acc_data, gait_predictions_all_folds, gait_labels_all_folds,
                    chorea_predictions_all_folds, chorea_labels_all_folds, valid_chorea_all_folds, valid_gait_all_folds,win_video_time):
    try:
        for fold_num, fold_ind in enumerate(cv_test_idxs_all_folds):
            for index_in_fold, ind in enumerate(fold_ind[0]):
                acc_data = win_acc_data[ind]
                subject = win_subjects[ind]
                video_time = win_video_time[ind]
                pred = gait_predictions_all_folds[ind]
                label = gait_labels_all_folds[ind]
                chorea_label = chorea_labels_all_folds[ind]
                valid_gait = valid_gait_all_folds[ind]
                acc_power = np.sqrt(np.mean(acc_data**2, axis=-1))
                acc_power_norm = acc_power/np.max(acc_power)
                plt.plot(acc_power_norm)
                plt.plot(label*0.9)
                plt.plot(pred)
                plt.plot(valid_gait)
                plt.plot(chorea_label/4)
                walking_ratio = np.sum(label*valid_gait)/(np.sum(valid_gait) + 1e-6)
                agreement_ratio = np.sum((pred==label)*valid_gait)/(np.sum(valid_gait) + 1e-6)
                plt.legend(['acc', 'label', 'pred', 'valid_gait','chorea_label'])
                plt.title(f'agreement ratio {agreement_ratio:.2f}')
                path_to_save = os.path.join(VIZUALIZE_DIR, "segmentation_visualize", f'{ind}_{subject}_{walking_ratio:.2f}_time_in_video_{video_time[0]:.2f}.jpg')
                print(f'saving {path_to_save}')
                plt.savefig(path_to_save)
                plt.close("all")
    except:
        ipdb.set_trace()