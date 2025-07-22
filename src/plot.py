import numpy as np
import matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
import pandas as pd
import argparse, os
from scipy.stats import ks_2samp, shapiro, mannwhitneyu, kstest

from generic_functions import load_dataset

def get_mse_psnr(a, b):
    max_ab = float(np.nanmax(np.concatenate((a, b))))
    mse = np.nanmean((np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)) ** 2)
    if mse == 0:
        return 0, 100
    return mse, 20 * np.log10(max_ab / (np.sqrt(mse)))


def retrun_indices(file):
    dataframe = pd.read_csv(file)
    indices = []
    [indices.append(row['Image']) for _, row in dataframe.iterrows()]
    return indices

def evaluate_cross_snr(indices, original_dataset, perturbed_dataset_h5, perturbed_dataset_tflite, perturbed_dataset_axc):
    h5_snr_list = []
    tflite_snr_list = []
    axc_snr_list = []
    
    for i in indices:
        _, h5_snr = get_mse_psnr(original_dataset[i], perturbed_dataset_h5[i])
        h5_snr_list.append(h5_snr)
        _, tflite_snr = get_mse_psnr(original_dataset[i], perturbed_dataset_tflite[i])
        tflite_snr_list.append(tflite_snr)
        _, axc_snr = get_mse_psnr(original_dataset[i], perturbed_dataset_axc[i])
        axc_snr_list.append(axc_snr)


    # _, p_value = ks_2samp(h5_snr_list, tflite_snr_list)
    # _, p_value2 = ks_2samp(tflite_snr_list, axc_snr_list)
    # _, p_value3 = ks_2samp(h5_snr_list, axc_snr_list)
    # print(f"SNR h5 vs tflite p-value: {p_value}")
    # print(f"SNR tflite vs axc p-value: {p_value2}")
    # print(f"SNR h5 vs axc p-value: {p_value3}")

    # if all(p > 0.05 for p in [p_value, p_value2, p_value3]):
    #     print("Non possiamo rifiutare l'ipotesi nulla: le distribuzioni sono simili.")
    # else:
    #     print("Rifiutiamo l'ipotesi nulla: almeno una distribuzione è diversa.")

        
    return h5_snr_list, tflite_snr_list, axc_snr_list


def psnr_wiscker_plot(h5_data, tflite_data, axc_data, figure):
    psnr_h5 =       pd.DataFrame({"CNN" :      h5_data})        
    psnr_tflite =   pd.DataFrame({"QNN" :     tflite_data})
    psnr_axc =      pd.DataFrame({"AxNN" :  axc_data})

    data = pd.concat([psnr_h5, psnr_tflite, psnr_axc])  
    mdf = pd.melt(data)

    if os.path.exists(figure):
        os.remove(figure)

    fig = plt.figure(figsize=[6, 4])
    ax = sns.boxplot(data = mdf, x = "variable", y = "value", width = 0.3, medianprops={'color': 'red', 'label': '_median_'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel("PSNR", fontsize = 20)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(figure)
    return

def psnr_statistics(h5_data, tflite_data, axc_data, psnr_stat_file):
    _, p_h5 = kstest(h5_data, 'norm')
    _, p_tflite = kstest(tflite_data, 'norm')
    _, p_axc = kstest(axc_data, 'norm')
      
    with open(psnr_stat_file, 'w') as res:
        if p_h5 > 0.05:
            res.write(f'PSNR h5 network is normally distributed with p-value {p_h5}\n')
        else:
            res.write(f'PSNR h5 network is not normally distributed with p-value {p_h5}\n')
            
        if p_tflite > 0.05:
            res.write(f'PSNR tflite network is normally distributed with p-value {p_tflite}\n')
        else:
            res.write(f'PSNR tflite network is not normally distributed with p-value {p_tflite}\n')
            
        if p_axc > 0.05:
            res.write(f'PSNR axc network is normally distributed with p-value {p_axc}\n')
        else:
            res.write(f'PSNR axc network is not normally distributed with p-value {p_axc}\n')
            
        _, p_h5_tflite = mannwhitneyu(h5_data, tflite_data)
        _, p_h5_axc = mannwhitneyu(h5_data, axc_data)
        _, p_tflite_axc = mannwhitneyu(tflite_data, axc_data)
        
        if p_h5_tflite > 0.05:
            res.write(f'PSNR h5 and tflite networks are similar with p-value {p_h5_tflite}\n')
        else:
            res.write(f'PSNR h5 and tflite networks are not similar with p-value {p_h5_tflite}\n')
            
        if p_h5_axc > 0.05:
            res.write(f'PSNR h5 and axc networks are similar with p-value {p_h5_axc}\n')
        else:
            res.write(f'PSNR h5 and axc networks are not similar with p-value {p_h5_axc}\n')
            
        if p_tflite_axc > 0.05:
            res.write(f'PSNR tflite and axc networks are similar with p-value {p_tflite_axc}\n')
        else:
            res.write(f'PSNR tflite and axc networks are not similar with p-value {p_tflite_axc}\n')


def wisker_plot_whiteBox_stats(csv_data_file, stats_file):
    dataframe = pd.read_csv(csv_data_file)
    h5_iterations =       []    
    tflite_iterations =   []
    axc_iterations =      []
    h5_perturabtions =      []
    tflite_perturabtions =  []
    axc_perturabtions =     []
    
    [ h5_iterations.append(       row['Number of iterations for h5'])     for _, row in dataframe.iterrows()]  
    [ tflite_iterations.append(   row['Number of iterations for tflite']) for _, row in dataframe.iterrows()] 
    [ axc_iterations.append(      row['Number of iterations for axc'])    for _, row in dataframe.iterrows()]
    [ h5_perturabtions.append(       np.round(row['Perturbation for h5']     * 255))     for _, row in dataframe.iterrows()]  
    [ tflite_perturabtions.append(   np.round(row['Perturbation for tflite'] * 255))     for _, row in dataframe.iterrows()]
    [ axc_perturabtions.append(      np.round(row['Perturbation for axc']    * 255))     for _, row in dataframe.iterrows()]
    
    with open(stats_file, 'w') as res:
        _, p_h5_iter = kstest(h5_iterations, 'norm')
        _, p_tflite_iter = kstest(tflite_iterations, 'norm')
        _, p_axc_iter = kstest(axc_iterations, 'norm')
        
        if p_h5_iter > 0.05:
            res.write(f'Iterations h5 network is normally distributed with p-value {p_h5_iter}\n')
        else:
            res.write(f'Iterations h5 network is not normally distributed with p-value {p_h5_iter}\n')
        
        if p_tflite_iter > 0.05:
            res.write(f'Iterations tflite network is normally distributed with p-value {p_tflite_iter}\n')
        else:
            res.write(f'Iterations tflite network is not normally distributed with p-value {p_tflite_iter}\n')
            
        if p_axc_iter > 0.05:
            res.write(f'Iterations axc network is normally distributed with p-value {p_axc_iter}\n')
        else:
            res.write(f'Iterations axc network is not normally distributed with p-value {p_axc_iter}\n')
            
        _, p_h5_tflite_iter = mannwhitneyu(h5_iterations, tflite_iterations)
        _, p_h5_axc_iter = mannwhitneyu(h5_iterations, axc_iterations)
        _, p_tflite_axc_iter = mannwhitneyu(tflite_iterations, axc_iterations)
        
        if p_h5_tflite_iter > 0.05:
           res.write(f'Iterations h5 and tflite networks are similar with p-value {p_h5_tflite_iter}\n')
        else:
            res.write(f'Iterations h5 and tflite networks are not similar with p-value {p_h5_tflite_iter}\n')
            
        if p_h5_axc_iter > 0.05:
            res.write(f'Iterations h5 and axc networks are similar with p-value {p_h5_axc_iter}\n')
        else:
            res.write(f'Iterations h5 and axc networks are not similar with p-value {p_h5_axc_iter}\n')
            
        if p_tflite_axc_iter > 0.05:
            res.write(f'Iterations tflite and axc networks are similar with p-value {p_tflite_axc_iter}\n')
        else:
            res.write(f'Iterations tflite and axc networks are not similar with p-value {p_tflite_axc_iter}\n')
            
        _, p_h5_pert = kstest(h5_perturabtions, 'norm')
        _, p_tflite_pert = kstest(tflite_perturabtions, 'norm')
        _, p_axc_pert = kstest(axc_perturabtions, 'norm')
        
        if p_h5_pert > 0.05:
            res.write(f'Perturbation h5 network is normally distributed with p-value {p_h5_pert}\n')
        else:
            res.write(f'Perturbation h5 network is not normally distributed with p-value {p_h5_pert}\n')
            
        if p_tflite_pert > 0.05:
            res.write(f'Perturbation tflite network is normally distributed with p-value {p_tflite_pert}\n')
        else:
            res.write(f'Perturbation tflite network is not normally distributed with p-value {p_tflite_pert}\n')
            
        if p_axc_pert > 0.05:
            res.write(f'Perturbation axc network is normally distributed with p-value {p_axc_pert}\n')
        else:
            res.write(f'Perturbation axc network is not normally distributed with p-value {p_axc_pert}\n')
            
        _, p_h5_tflite_per = mannwhitneyu(h5_perturabtions, tflite_perturabtions)
        _, p_h5_axc_per = mannwhitneyu(h5_perturabtions, axc_perturabtions)
        _, p_tflite_axc_per = mannwhitneyu(tflite_perturabtions, axc_perturabtions)
        
        if p_h5_tflite_per > 0.05:
            res.write(f'Perturbation h5 and tflite networks are similar with p-value {p_h5_tflite_per}\n')
        else:
            res.write(f'Perturbation h5 and tflite networks are not similar with p-value {p_h5_tflite_per}\n')
            
        if p_h5_axc_per > 0.05:
            res.write(f'Perturbation h5 and axc networks are similar with p-value {p_h5_axc_per}\n')
        else:
            res.write(f'Perturbation h5 and axc networks are not similar with p-value {p_h5_axc_per}\n')
            
        if p_tflite_axc_per > 0.05:
            res.write(f'Perturbation tflite and axc networks are similar with p-value {p_tflite_axc_per}\n')
        else:
            res.write(f'Perturbation tflite and axc networks are not similar with p-value {p_tflite_axc_per}\n')
    
    

def wisker_plot_whiteBox(csv_data_file, iteration_igure, perturbation_figure):

    dataframe = pd.read_csv(csv_data_file)
    h5_iterations =       []    
    tflite_iterations =   []
    axc_iterations =      []
    h5_perturabtions =      []
    tflite_perturabtions =  []
    axc_perturabtions =     []
    
    [ h5_iterations.append(       row['Number of iterations for h5'])     for _, row in dataframe.iterrows()]  
    [ tflite_iterations.append(   row['Number of iterations for tflite']) for _, row in dataframe.iterrows()] 
    [ axc_iterations.append(      row['Number of iterations for axc'])    for _, row in dataframe.iterrows()]
    [ h5_perturabtions.append(       np.round(row['Perturbation for h5']     * 255))     for _, row in dataframe.iterrows()]  
    [ tflite_perturabtions.append(   np.round(row['Perturbation for tflite'] * 255))     for _, row in dataframe.iterrows()]
    [ axc_perturabtions.append(      np.round(row['Perturbation for axc']    * 255))     for _, row in dataframe.iterrows()] 

    it_h5 =       pd.DataFrame({"CNN" :     h5_iterations })        
    it_tflite =   pd.DataFrame({"QNN" : tflite_iterations})
    it_axc =      pd.DataFrame({"AxNN" :    axc_iterations})
    pert_h5 =     pd.DataFrame({"CNN" :     h5_perturabtions})        
    pert_tflite = pd.DataFrame({"QNN" : tflite_perturabtions})
    pert_axc =    pd.DataFrame({"AxNN" :    axc_perturabtions})
    it_data = pd.concat([it_h5, it_tflite, it_axc])
    pert_data = pd.concat([pert_h5, pert_tflite, pert_axc])
    it_mdf = pd.melt(it_data)
    pert_mdf = pd.melt(pert_data)

    if os.path.exists(iteration_igure):
        os.remove(iteration_igure)

    if os.path.exists(perturbation_figure):
        os.remove(perturbation_figure)
        
    # psnr_h5 =       pd.DataFrame({"Original DNN" :      h5_data})        
    # psnr_tflite =   pd.DataFrame({"Quantized DNN" :     tflite_data})
    # psnr_axc =      pd.DataFrame({"Approximated DNN" :  axc_data})

    # data = pd.concat([psnr_h5, psnr_tflite, psnr_axc])  
    # mdf = pd.melt(data)

    fig1 = plt.figure(figsize=[6, 4])
    ax = sns.boxplot(data = it_mdf, x = "variable", y = "value", width = 0.3, medianprops={'color': 'red', 'label': '_median_'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel("Number of Iterations", fontsize = 20)
    plt.xlabel('DNNs', fontsize = 20)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(iteration_igure)
    plt.close()

    fig2 = plt.figure(figsize=[6, 4])
    ax1 = sns.boxplot(data = pert_mdf, x = "variable", y = "value", width = 0.3, medianprops={'color': 'red', 'label': '_median_'})
    ax1.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax1.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel("Perturbation Strenght", fontsize = 20)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(perturbation_figure)
    plt.close()
 

# def effort_white(csv_data_file, figure):
#     dataframe = pd.read_csv(csv_data_file)
#     hist_tflite = {}
#     hist_axc = {}
    
#     for _, row in dataframe.iterrows():
#     # if row['Perturbation for h5'] == row['Perturbation for tflite']:
#         if row['Number of iterations for tflite'] - row['Number of iterations for h5'] not in hist_tflite.keys():
#             hist_tflite[row['Number of iterations for tflite'] - row['Number of iterations for h5']] = []
#         hist_tflite[row['Number of iterations for tflite'] - row['Number of iterations for h5']].append(row['Image'])

#     # if row['Perturbation for h5'] == row['Perturbation for axc']:
#         if row['Number of iterations for axc'] - row['Number of iterations for h5'] not in hist_axc.keys():
#             hist_axc[row['Number of iterations for axc'] - row['Number of iterations for h5']] = []
#         hist_axc[row['Number of iterations for axc'] - row['Number of iterations for h5']].append(row['Image'])

#     x_values = sorted(set(list(hist_tflite.keys()) + list(hist_axc.keys())))
#     x = list(range(min(x_values), max(x_values) + 1))
    
    
#     y_tflite = [len(hist_tflite[i]) if i in hist_tflite.keys() else 0 for i in x]
#     y_axc = [len(hist_axc[i]) if i in hist_axc.keys() else 0 for i in x]

#     if os.path.exists(figure):
#         os.remove(figure)
            
#     fig = plt.figure(figsize=[6, 4])
#     plt.bar(np.array(x),     y_tflite, width = 0.4, align="edge", label = "Quantized NN")
#     plt.bar(np.array(x)+0.4, y_axc, width = 0.4, align="edge", label = "Approximated NN")
#     plt.xlabel("Number of Iterations more than H5")
#     plt.ylabel("Number of Images")
#     plt.xticks(x, rotation = 90, fontsize = 10)
#     plt.legend(frameon = False)
#     plt.grid(visible=True)
#     plt.tight_layout()
#     plt.savefig(figure)
#     plt.close()
    
    

def transfereability_data_white(csv_data_file, results_data):
    dataframe = pd.read_csv(csv_data_file)
    tlite_same_iteraions = 0
    tflite_same_perturbation = 0
    tflite_all_different = 0
    axc_same_iterations = 0
    axc_same_perturbation = 0
    axc_all_different = 0
    tflite_transferability = 0
    axc_transferability = 0
    tflite_iterations_difference = []
    axc_iterations_difference = []
    tflite_perturbation_difference = []
    axc_perturbation_difference = []
    for _, row in dataframe.iterrows():
        if row['Perturbation for h5'] == row['Perturbation for tflite'] and row['Number of iterations for h5'] == row['Number of iterations for tflite']:
            tflite_transferability += 1
        if row['Perturbation for h5'] == row['Perturbation for tflite'] and not row['Number of iterations for h5'] == row['Number of iterations for tflite']:
            tflite_same_perturbation += 1
            tflite_iterations_difference.append(row['Number of iterations for tflite'] - row['Number of iterations for h5'])
        if not row['Perturbation for h5'] == row['Perturbation for tflite'] and row['Number of iterations for h5'] == row['Number of iterations for tflite']:
            tlite_same_iteraions += 1
            tflite_perturbation_difference.append((row['Perturbation for tflite'] - row['Perturbation for h5']))
        if not row['Perturbation for h5'] == row['Perturbation for tflite'] and not row['Number of iterations for h5'] == row['Number of iterations for tflite']:
            tflite_all_different += 1
            
        if row['Perturbation for h5'] == row['Perturbation for axc'] and row['Number of iterations for h5'] == row['Number of iterations for axc']:
            axc_transferability += 1
        if row['Perturbation for h5'] == row['Perturbation for axc'] and not row['Number of iterations for h5'] == row['Number of iterations for axc']:
            axc_same_perturbation += 1
            axc_iterations_difference.append(row['Number of iterations for axc'] - row['Number of iterations for h5'])
        if not row['Perturbation for h5'] == row['Perturbation for axc'] and row['Number of iterations for h5'] == row['Number of iterations for axc']:
            axc_same_iterations += 1
            axc_perturbation_difference.append((row['Perturbation for axc'] - row['Perturbation for h5']))
        if not row['Perturbation for h5'] == row['Perturbation for axc'] and not row['Number of iterations for h5'] == row['Number of iterations for axc']:
            axc_all_different += 1

    with open(results_data, 'w') as res:
        res.write(f'Total images: {len(dataframe.index)}\n')
        res.write(f'H5 tranferability to tflite {tflite_transferability}\nSame perturbation {tflite_same_perturbation}. Mean iterations-difference: {np.mean(tflite_iterations_difference)}. Median iterations-difference: {np.median(tflite_iterations_difference)}. Same iterations {tlite_same_iteraions}. Mean pertubation-difference: {np.mean(tflite_perturbation_difference) * 255}. Median pertubation-difference: {np.median(tflite_perturbation_difference) * 255}. All different {tflite_all_different}\n')
        res.write(f'H5 tranferability to AxC {axc_transferability}\nSame perturbation {axc_same_perturbation}. Mean iterations-difference: {np.mean(axc_iterations_difference)}. Median iterations-difference: {np.median(axc_iterations_difference)}. Same iterations {axc_same_iterations}. Mean pertubation-difference: {np.mean(axc_perturbation_difference) * 255}. Median pertubation-difference: {np.median(axc_perturbation_difference) * 255}. All different {axc_all_different}')


def wisker_plot_blackBox_stats(csv_data_file, stats_file):
    dataframe = pd.read_csv(csv_data_file)
    h5_data =       []    
    tflite_data =   []
    axc_data =      []
     
    for _, row in dataframe.iterrows():
        if row['h5_fooled'] == True:
            h5_data.append(row['nit_h5'])
        if row['tflite_fooled'] == True:
            tflite_data.append(row['nit_tflite'])
        if row['axc_fooled'] == True:
            axc_data.append(row['nit_axc'])
            
    with open(stats_file, 'w') as res:
        _, p_h5 = kstest(h5_data, 'norm')
        _, p_tflite = kstest(tflite_data, 'norm')
        _, p_axc = kstest(axc_data, 'norm')
        
        if p_h5 > 0.05:
            res.write(f'Iterations h5 network is normally distributed with p-value {p_h5}\n')
        else:
            res.write(f'Iterations h5 network is not normally distributed with p-value {p_h5}\n')
            
        if p_tflite > 0.05:
            res.write(f'Iterations tflite network is normally distributed with p-value {p_tflite}\n')
        else:
            res.write(f'Iterations tflite network is not normally distributed with p-value {p_tflite}\n')
            
        if p_axc > 0.05:
            res.write(f'Iterations axc network is normally distributed with p-value {p_axc}\n')
        else:
            res.write(f'Iterations axc network is not normally distributed with p-value {p_axc}\n')
            
        _, p_h5_tflite = mannwhitneyu(h5_data, tflite_data)
        _, p_h5_axc = mannwhitneyu(h5_data, axc_data)
        _, p_tflite_axc = mannwhitneyu(tflite_data, axc_data)
        
        if p_h5_tflite > 0.05:
              res.write(f'Iterations h5 and tflite networks are similar with p-value {p_h5_tflite}\n')
        else:
            res.write(f'Iterations h5 and tflite networks are not similar with p-value {p_h5_tflite}\n')
            
        if p_h5_axc > 0.05:
            res.write(f'Iterations h5 and axc networks are similar with p-value {p_h5_axc}\n')
        else:
            res.write(f'Iterations h5 and axc networks are not similar with p-value {p_h5_axc}\n')
            
        if p_tflite_axc > 0.05:
            res.write(f'Iterations tflite and axc networks are similar with p-value {p_tflite_axc}\n')
        else:
            res.write(f'Iterations tflite and axc networks are not similar with p-value {p_tflite_axc}\n')


def wisker_plot_blackBox(csv_data_file, figure):
    dataframe = pd.read_csv(csv_data_file)
    h5_data =       []    
    tflite_data =   []
    axc_data =      []
     
    for _, row in dataframe.iterrows():
        if row['h5_fooled'] == True:
            h5_data.append(row['nit_h5'])
        if row['tflite_fooled'] == True:
            tflite_data.append(row['nit_tflite'])
        if row['axc_fooled'] == True:
            axc_data.append(row['nit_axc'])

            
    h5 =        pd.DataFrame({"CNN" :  h5_data    })        
    tflite =    pd.DataFrame({"QNN" :  tflite_data})
    axc =       pd.DataFrame({"AxNN" : axc_data   })
    data = pd.concat([h5, tflite, axc])   
    mdf = pd.melt(data)

    if os.path.exists(figure):
        os.remove(figure)
    
    fig = plt.figure(figsize=[6, 4])
    ax = sns.boxplot(data=mdf, x="variable", y="value", width=0.4, medianprops={'color': 'red', 'label': '_median_'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel("Number of Generations", fontsize = 20)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()       


def transfereability_data_black(csv_data_file, result_data):
    dataframe = pd.read_csv(csv_data_file)
    h5_fooled = 0
    tflite_fooled = 0
    axc_fooled = 0
    tflite_fooled_byh5 = 0
    axc_fooled_byh5 = 0
    axc_fooled_bytflite = 0
    axc_to_h5 = 0  
      
    for _, row in dataframe.iterrows():
        if row['h5_fooled']:
            h5_fooled = h5_fooled + 1
            if row['h5_to_tflite']:
                tflite_fooled_byh5 = tflite_fooled_byh5 + 1
            if row['h5_to_axc']:
                axc_fooled_byh5 = axc_fooled_byh5 + 1
        if row['tflite_fooled']:
            tflite_fooled = tflite_fooled + 1
            if row['tflite_to_axc']:
                axc_fooled_bytflite += 1
        if row['axc_fooled']:
            axc_fooled += 1
            if row['axc_to_h5_back']:
                axc_to_h5 += 1
                
    with open(result_data, 'w') as res:
        res.write(f'Total images: {len(dataframe.index)} \nH5 fooled {h5_fooled} times\nTflite fooled {tflite_fooled_byh5} times by h5\nAXC fooled {axc_fooled_byh5} times by h5\n')
        res.write(f'Tflite fooled {tflite_fooled} times\nAXC fooled {axc_fooled_bytflite} times by tflite\n')
        res.write(f'AxC fooled {axc_fooled} times\nH5 fooled {axc_to_h5} times by AxC')
        

# def effort_black(csv_data_file, figure):
#     dataframe = pd.read_csv(csv_data_file)
#     hist_tflite = {}
#     hist_axc = {}
    
#     for _, row in dataframe.iterrows():
#         if row['h5_fooled'] and row['tflite_fooled'] and not row['h5_to_tflite']:
#             if row['nit_tflite'] - row['nit_h5'] not in hist_tflite.keys():
#                 hist_tflite[row['nit_tflite'] - row['nit_h5']] = []
#             hist_tflite[row['nit_tflite'] - row['nit_h5']].append(row['Image'])
            
#         if row['h5_fooled'] and row['axc_fooled'] and not row['h5_to_axc']:
#             if row['nit_axc'] - row['nit_h5'] not in hist_axc.keys():
#                 hist_axc[row['nit_axc'] - row['nit_h5']] = []
#             hist_axc[row['nit_axc'] - row['nit_h5']].append(row['Image'])

#     x_values = sorted(set(list(hist_tflite.keys()) + list(hist_axc.keys())))
#     x = list(range(min(x_values), max(x_values) + 1))
    
    
#     y_tflite = [len(hist_tflite[i]) if i in hist_tflite.keys() else 0 for i in x]
#     y_axc = [len(hist_axc[i]) if i in hist_axc.keys() else 0 for i in x]
            
#     if os.path.exists(figure):
#         os.remove(figure)

#     fig = plt.figure(figsize=[6, 4])
#     plt.bar(np.array(x),     y_tflite, width = 0.4, align="edge", label = "Quantized NN")
#     plt.bar(np.array(x)+0.4, y_axc, width = 0.4, align="edge", label = "Approximated NN")
#     plt.xlabel("Number of Generations more than H5")
#     plt.ylabel("Number of Images")
#     plt.xticks(x, rotation = 90, fontsize = 10)
#     plt.legend(frameon = False)
#     plt.grid(visible=True)
#     plt.tight_layout()
#     plt.savefig(figure)
#     plt.close()
#     return
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",  "-p", type = str, help = "Repository with results",                                  required = True)
    parser.add_argument("--model", "-m", type = str, help = "Model to attack. Please specify also it's relative path.", required = True)
    args = parser.parse_args()
    
    model_name =  args.model.split("/")[-1].split(".")[0]
    file =                              os.path.join(args.path, "results.csv")
    transferability_data =              os.path.join(args.path, "transferability_data.txt")
    effort_figure =                     os.path.join(args.path, "effort.pdf")
    perturbed_dataset_h5 =      np.load(os.path.join(args.path, "h5_images.npy"))
    perturbed_dataset_tflite =  np.load(os.path.join(args.path, "tflite_images.npy"))
    perturbed_dataset_axc =     np.load(os.path.join(args.path, "axc_images.npy"))
    
    model = tf.keras.models.load_model(args.model)
    # model_name = args.model.split("/")[-1].split(".")[0]
    # _, _, _, _, original_dataset, _, _ = load_dataset(model.input_shape, model_name)
    _, _, _, _, original_dataset, _, _ = load_dataset((None, 32, 32, 3), model_name)
    indices = retrun_indices(file)

    
    h5_snr_list, tflite_snr_list, axc_snr_list = evaluate_cross_snr(indices, original_dataset, perturbed_dataset_h5, perturbed_dataset_tflite, perturbed_dataset_axc)
    

    iteration_figure = os.path.join(args.path, "iterations_" + model_name + ".pdf")
    perturbation_figure = os.path.join(args.path, "perturbation_" + model_name + ".pdf")
    
    psnr_statistics(h5_snr_list, tflite_snr_list, axc_snr_list, os.path.join(args.path, "PSNRstats.txt"))
  
    # psnr_wiscker_plot(h5_snr_list, tflite_snr_list, axc_snr_list, os.path.join(args.path, "psnr_" + model_name+ ".pdf"))
    if "WhiteBox" in args.path:
        # wisker_plot_whiteBox(file, iteration_figure, perturbation_figure)
        # transfereability_data_white(file, transferability_data)
        wisker_plot_whiteBox_stats(file, os.path.join(args.path, "stats.txt"))
    elif "BlackBox" in args.path:
        #wisker_plot_blackBox(file , iteration_figure)
        # transfereability_data_black(file, transferability_data)
        wisker_plot_blackBox_stats(file, os.path.join(args.path, "stats.txt"))

