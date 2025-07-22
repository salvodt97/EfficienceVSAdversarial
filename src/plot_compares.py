import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd, argparse, os
from scipy.stats import shapiro, mannwhitneyu

from generic_functions import load_dataset, create_repos

def get_mse_psnr(a, b):
    max_ab = float(np.nanmax(np.concatenate((a, b))))
    mse = np.nanmean((np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)) ** 2)
    if mse == 0:
        return 0, 100
    return mse, 20 * np.log10(max_ab / (np.sqrt(mse)))


def return_indices(file1, file2):
    dataframe1 = pd.read_csv(file1)
    dataframe2 = pd.read_csv(file2)
    indices1 = set(dataframe1['Image'])
    indices2 = set(dataframe2['Image'])
    common_indices = list(indices1 & indices2)
    return common_indices


def evaluate_cross_snr(indices, original_dataset, perturbed_dataset_shallower, perturbed_dataset_deeper):
    snr_list_shallower = []
    snr_list_deeper = []
    
    for i in indices:
        _, shallower_snr = get_mse_psnr(original_dataset[i], perturbed_dataset_shallower[i])
        snr_list_shallower.append(shallower_snr)
        _, deeper_snr = get_mse_psnr(original_dataset[i], perturbed_dataset_deeper[i])
        snr_list_deeper.append(deeper_snr)

    return snr_list_shallower, snr_list_deeper        


def cross_psnr_wisker_plot(snr_list_shallower, snr_list_deeper, model1_name, model2_name, figure, type):
    psnr_shallower =       pd.DataFrame({model1_name:      snr_list_shallower})
    psnr_deeper =          pd.DataFrame({model2_name:      snr_list_deeper})

    data = pd.concat([psnr_shallower, psnr_deeper])
    mdf = pd.melt(data)

    if os.path.exists(figure):
        os.remove(figure)

    fig = plt.figure(figsize=[6, 4])
    ax = sns.boxplot(data = mdf, x = "variable", y = "value", width = 0.3, medianprops={'color': 'red', 'label': '_median_'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel("PSNR", fontsize = 20)
    plt.xlabel(type, fontsize = 20)
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()
    
def cross_psnr_stats_test(snr_list_shallower, snr_list_deeper, model1_name, model2_name, type, file_stats):
    with open(file_stats, 'a') as res:
        _, p_value_shallower = shapiro(snr_list_shallower)
        _, p_value_deeper = shapiro(snr_list_deeper)
        
        if p_value_shallower < 0.05:
            res.write(f"Shapiro: {model1_name} PSNR for {type} is not normally distributed (p-value: {p_value_shallower})\n")
        else:
            res.write(f"Shapiro: {model1_name} PSNR for {type} is normally distributed (p-value: {p_value_shallower})\n")
            
        if p_value_deeper < 0.05:
            res.write(f"Shapiro: {model2_name} PSNR for {type} is not normally distributed (p-value: {p_value_deeper})\n")
        else:
            res.write(f"Shapiro: {model2_name} PSNR for {type} is normally distributed (p-value: {p_value_deeper})\n")
            
        _, p_value = mannwhitneyu(snr_list_shallower, snr_list_deeper)
        if p_value > 0.05:
            res.write(f"Wilcoxon PSNR for {type} is not statistically significant (p-value: {p_value})\n")
        else:
            res.write(f"Wilcoxon PSNR for {type} is statistically significant (p-value: {p_value})\n")

def box_plot_whiteBox(indices, shallower_file, deeper_file, model1_name, model2_name, figure, network_type, dataset_type):

    dataframe_shallow = pd.read_csv(shallower_file)
    dataframe_deeper = pd.read_csv(deeper_file)

    shallower_list = []
    deeper_list = []

    if network_type == 'Original CNNs':
        type = 'h5'
    elif network_type == 'Quantized CNNs':
        type = 'tflite'
    elif network_type == 'Approximate CNNs':
        type = 'axc'
    else:
        raise ValueError('Network type not recognized')

    [shallower_list.append(row[dataset_type +' for ' + type]) for _, row in dataframe_shallow.iterrows() if row['Image'] in indices]
    [deeper_list.append(row[dataset_type +' for ' + type]) for _, row in dataframe_deeper.iterrows() if row['Image'] in indices]
 
    shallower = pd.DataFrame({model1_name: shallower_list})
    deeper =    pd.DataFrame({model2_name: deeper_list})
    data = pd.concat([shallower, deeper])
    mdf = pd.melt(data)

    if os.path.exists(figure):
        os.remove(figure)

    fig = plt.figure(figsize=[6, 4])
    ax = sns.boxplot(data = mdf, x = "variable", y = "value", width = 0.3, medianprops={'color': 'red', 'label': '_median_'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel(dataset_type, fontsize = 20)
    plt.xlabel(network_type, fontsize = 20)
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()
    
def whiteBox_stats_test(indices, shallower_file, deeper_file, model1_name, model2_name, network_type, dataset_type, file_stats):
    dataframe_shallow = pd.read_csv(shallower_file)
    dataframe_deeper = pd.read_csv(deeper_file)

    shallower_list = []
    deeper_list = []

    if network_type == 'Original CNNs':
        type = 'h5'
    elif network_type == 'Quantized CNNs':
        type = 'tflite'
    elif network_type == 'Approximate CNNs':
        type = 'axc'
    else:
        raise ValueError('Network type not recognized')

    [shallower_list.append(row[dataset_type +' for ' + type]) for _, row in dataframe_shallow.iterrows() if row['Image'] in indices]
    [deeper_list.append(row[dataset_type +' for ' + type]) for _, row in dataframe_deeper.iterrows() if row['Image'] in indices]
    
    with open(file_stats, 'a') as res:
        _, p_value_shallower = shapiro(shallower_list)
        _, p_value_deeper = shapiro(deeper_list)
        
        if p_value_shallower < 0.05:
            res.write(f"Shapiro: {model1_name} {dataset_type} for {type} is not normally distributed (p-value: {p_value_shallower})\n")
        else:
            res.write(f"Shapiro: {model1_name} {dataset_type} for {type} is normally distributed (p-value: {p_value_shallower})\n")
            
        if p_value_deeper < 0.05:
            res.write(f"Shapiro: {model2_name} {dataset_type} for {type} is not normally distributed (p-value: {p_value_deeper})\n")
        else:
            res.write(f"Shapiro: {model2_name} {dataset_type} for {type} is normally distributed (p-value: {p_value_deeper})\n")
            
        _, p_value = mannwhitneyu(shallower_list, deeper_list)
        if p_value > 0.05:
            res.write(f"Wilcoxon difference {dataset_type} for {type} is not statistically significant (p-value: {p_value})\n")
        else:
            res.write(f"Wilcoxon difference {dataset_type} for {type} is statistically significant (p-value: {p_value})\n")


def wisker_plot_blackBox(indices, shallower_file, deeper_file, model1_name, model2_name, figure, network_type):
    dataframe_shallower = pd.read_csv(shallower_file)
    dataframe_deeper = pd.read_csv(deeper_file)

    shallower_list = []
    deeper_list = []

    if network_type == 'Original CNNs':
        type = 'h5'
    elif network_type == 'Quantized CNNs':
        type = 'tflite'
    elif network_type == 'Approximate CNNs':
        type = 'axc'
    else:
        raise ValueError('Network type not recognized')
     
    [shallower_list.append(row['nit_' + type]) for _, row in dataframe_shallower.iterrows() if row['Image'] in indices and row[type + '_fooled'] == True]
    [deeper_list.append(row['nit_' + type]) for _, row in dataframe_deeper.iterrows() if row['Image'] in indices and row[type + '_fooled'] == True]

    shallower = pd.DataFrame({model1_name: shallower_list})
    eeper =    pd.DataFrame({model2_name: deeper_list})
    data = pd.concat([shallower, deeper])
    mdf = pd.melt(data)
            
    if os.path.exists(figure):
        os.remove(figure)
    
    fig = plt.figure(figsize=[6, 4])
    ax = sns.boxplot(data=mdf, x="variable", y="value", width=0.4, medianprops={'color': 'red', 'label': '_median_'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 20, rotation = 45)
    plt.ylabel("Number of Generations", fontsize = 20)
    plt.xlabel(network_type, fontsize = 20)
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()
    
    
def blackBox_stats_test(indices, shallower_file, deeper_file, model1_name, model2_name, network_type, file_stat):    
    dataframe_shallower = pd.read_csv(shallower_file)
    dataframe_deeper = pd.read_csv(deeper_file)

    shallower_list = []
    deeper_list = []

    if network_type == 'Original CNNs':
        type = 'h5'
    elif network_type == 'Quantized CNNs':
        type = 'tflite'
    elif network_type == 'Approximate CNNs':
        type = 'axc'
    else:
        raise ValueError('Network type not recognized')
     
    [shallower_list.append(row['nit_' + type]) for _, row in dataframe_shallower.iterrows() if row['Image'] in indices and row[type + '_fooled'] == True]
    [deeper_list.append(row['nit_' + type]) for _, row in dataframe_deeper.iterrows() if row['Image'] in indices and row[type + '_fooled'] == True]
    
    with open(file_stat, 'a') as res:
        _, p_value_shallower = shapiro(shallower_list)
        _, p_value_deeper = shapiro(deeper_list)
        
        if p_value_shallower < 0.05:
            res.write(f"Shapiro: {model1_name} {type} Number of iterations is not normally distributed (p-value: {p_value_shallower})\n")
        else:
            res.write(f"Shapiro: {model1_name} {type} Number of iterations is normally distributed (p-value: {p_value_shallower})\n")
            
        if p_value_deeper < 0.05:
            res.write(f"Shapiro: {model2_name} {type} Number of iterations is not normally distributed (p-value: {p_value_deeper})\n")
        else:
            res.write(f"Shapiro: {model2_name} {type} Number of iterations is normally distributed (p-value: {p_value_deeper})\n")
            
        _, p_value = mannwhitneyu(shallower_list, deeper_list)
        if p_value > 0.05:
            res.write(f"Wilcoxon difference in number of iterations for {type} is not statistically significant (p-value: {p_value})\n")
        else:
            res.write(f"Wilcoxon Difference in number of iterations for {type} is statistically significant (p-value: {p_value})\n")
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_shallower",  "-s", type = str, help = "Repository with results of shallower network", required = True)
    parser.add_argument("--path_deeper",     "-d", type = str, help = "Repository with results of shallower network", required = True)
    parser.add_argument("--path",            "-p", type = str, help = "Repository in which store results", required = True)
    parser.add_argument("--attack",          "-a", type = str, help = "Target Attack", required = True)
    args = parser.parse_args()

    # create_repos(args.path)
    
    model1_name =  args.path_shallower.split("/")[-1].split(".")[0].split("_")[0]
    model2_name =  args.path_deeper.split("/")[-1].split(".")[0].split("_")[0]
    file_shallower = os.path.join(args.path_shallower, "results.csv")
    file_deeper =    os.path.join(args.path_deeper, "results.csv")

    perturbed_dataset_h5_shallower =        np.load(os.path.join(args.path_shallower, "h5_images.npy"))
    perturbed_dataset_tflite_shallower =    np.load(os.path.join(args.path_shallower, "tflite_images.npy"))
    perturbed_dataset_axc_shallower =       np.load(os.path.join(args.path_shallower, "axc_images.npy"))
    perturbed_dataset_h5_deeper =           np.load(os.path.join(args.path_deeper, "h5_images.npy"))
    perturbed_dataset_tflite_deeper =       np.load(os.path.join(args.path_deeper, "tflite_images.npy"))
    perturbed_dataset_axc_deeper =          np.load(os.path.join(args.path_deeper, "axc_images.npy"))

    psnr_h5_figure = os.path.join(args.path, args.attack + "_h5_psnr.pdf")
    psnr_tflite_figure = os.path.join(args.path, args.attack + "_tflite_psnr.pdf")
    psnr_axc_figure = os.path.join(args.path, args.attack + "_axc_psnr.pdf")
    iteration_h5_figure = os.path.join(args.path, args.attack + "_iterations_h5.pdf")
    iteration_tflite_figure = os.path.join(args.path, args.attack + "_iterations_tflite.pdf")
    iteration_axc_figure = os.path.join(args.path, args.attack + "_iterations_axc.pdf")
    perturbation_h5_figure = os.path.join(args.path, args.attack + "_perturbation_h5.pdf")
    perturbation_tflite_figure = os.path.join(args.path, args.attack + "_perturbation_tflite.pdf")
    perturbation_axc_figure = os.path.join(args.path, args.attack + "_perturbation_axc.pdf")


    _, _, _, _, original_dataset, _, _ = load_dataset((None, 32, 32, 3), model1_name)
    indices = return_indices(file_shallower, file_deeper)
    
    h5_snr_list_shallower, h5_snr_list_deeper =         evaluate_cross_snr(indices, original_dataset, perturbed_dataset_h5_shallower, perturbed_dataset_h5_deeper)
    tflite_snr_list_shallower, tflite_snr_list_deeper = evaluate_cross_snr(indices, original_dataset, perturbed_dataset_tflite_shallower, perturbed_dataset_tflite_deeper)
    axc_snr_list_shallower, axc_snr_list_deeper =       evaluate_cross_snr(indices, original_dataset, perturbed_dataset_axc_shallower, perturbed_dataset_axc_deeper)
    
    
    #if os.path.exists(os.path.join(args.path, "PSNR_stats.txt")):
    #    os.remove(os.path.join(args.path, "PSNR_stats.txt"))
        
    #cross_psnr_stats_test(h5_snr_list_shallower, h5_snr_list_deeper, model1_name, model2_name, 'Original CNNs', os.path.join(args.path, "PSNR_stats.txt"))
    #cross_psnr_stats_test(tflite_snr_list_shallower, tflite_snr_list_deeper, model1_name, model2_name, 'Quantized CNNs', os.path.join(args.path, "PSNR_stats.txt"))
    #cross_psnr_stats_test(axc_snr_list_shallower, axc_snr_list_deeper, model1_name, model2_name, 'Approximate CNNs', os.path.join(args.path, "PSNR_stats.txt"))


    #cross_psnr_wisker_plot(h5_snr_list_shallower, h5_snr_list_deeper, model1_name, model2_name, psnr_h5_figure, 'Original CNNs')
    #cross_psnr_wisker_plot(tflite_snr_list_shallower, tflite_snr_list_deeper, model1_name, model2_name, psnr_tflite_figure, 'Quantized CNNs')
    #cross_psnr_wisker_plot(axc_snr_list_shallower, axc_snr_list_deeper, model1_name, model2_name, psnr_axc_figure, 'Approximate CNNs')
  
    if "WhiteBox" in args.path_shallower or "WhiteBox" in args.path_deeper:
        if os.path.exists(os.path.join(args.path, "stats.txt")):
            os.remove(os.path.join(args.path, "stats.txt"))
        box_plot_whiteBox(indices, file_shallower, file_deeper, model1_name, model2_name, iteration_h5_figure, 'Original CNNs', 'Number of iterations')
        box_plot_whiteBox(indices, file_shallower, file_deeper, model1_name, model2_name, iteration_tflite_figure, 'Quantized CNNs', 'Number of iterations')
        box_plot_whiteBox(indices, file_shallower, file_deeper, model1_name, model2_name, iteration_axc_figure, 'Approximate CNNs', 'Number of iterations')
        box_plot_whiteBox(indices, file_shallower, file_deeper, model1_name, model2_name, perturbation_h5_figure, 'Original CNNs', 'Perturbation')
        box_plot_whiteBox(indices, file_shallower, file_deeper, model1_name, model2_name, perturbation_tflite_figure, 'Quantized CNNs', 'Perturbation')
        box_plot_whiteBox(indices, file_shallower, file_deeper, model1_name, model2_name, perturbation_axc_figure, 'Approximate CNNs', 'Perturbation')
        whiteBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Original CNNs', 'Number of iterations', os.path.join(args.path, "stats.txt"))
        whiteBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Quantized CNNs', 'Number of iterations', os.path.join(args.path, "stats.txt"))
        whiteBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Approximate CNNs', 'Number of iterations', os.path.join(args.path, "stats.txt"))
        whiteBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Original CNNs', 'Perturbation', os.path.join(args.path, "stats.txt"))
        whiteBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Quantized CNNs', 'Perturbation', os.path.join(args.path, "stats.txt"))
        whiteBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Approximate CNNs', 'Perturbation', os.path.join(args.path, "stats.txt"))
    elif "BlackBox" in args.path_shallower or "BlackBox" in args.path_deeper:
        if os.path.exists(os.path.join(args.path, "stats.txt")):
            os.remove(os.path.join(args.path, "stats.txt"))
        #wisker_plot_blackBox(indices, file_shallower, file_deeper, model1_name, model2_name, iteration_h5_figure, 'Original CNNs')
        #wisker_plot_blackBox(indices, file_shallower, file_deeper, model1_name, model2_name, iteration_tflite_figure, 'Quantized CNNs')
        #wisker_plot_blackBox(indices, file_shallower, file_deeper, model1_name, model2_name, iteration_axc_figure, 'Approximate CNNs')
        blackBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Original CNNs', os.path.join(args.path, "stats.txt"))
        blackBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Quantized CNNs', os.path.join(args.path, "stats.txt"))
        blackBox_stats_test(indices, file_shallower, file_deeper, model1_name, model2_name, 'Approximate CNNs', os.path.join(args.path, "stats.txt"))


