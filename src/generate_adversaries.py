import os, csv, argparse, json
import numpy as np, matplotlib.pyplot  as plt, tensorflow as tf
import tensorflow_model_optimization as tfmot
import scipy.optimize as opt
from tqdm import tqdm
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import CarliniL2Method, DeepFool, ProjectedGradientDescent

from generic_functions import load_dataset, evaluate_nets, return_predicted, limit_resource_usage, take_quantization_parameters, approximate_model, resnet50_preprocess_images, classes_cif10, classes_mnist
from predict_general import general_predict

def iterative_fgsm(model, x, y, epsilon, num_iter):
    x = tf.convert_to_tensor(x)
    x_adv = x
    y = tf.convert_to_tensor(y)
    y = tf.expand_dims(y, axis=0)
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv)
            loss_value = tf.keras.losses.CategoricalCrossentropy()(y, logits)
        grads, = tape.gradient(loss_value, x_adv)
        x_adv = x_adv + epsilon * tf.sign(grads)
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
    return np.squeeze(x_adv, axis=0)


def evaluate_speedly(network, quantized_network, axc_model, dataset, results_path, h5_adv_vector, tflite_adv_vector, axc_adv_vector):
    csv_file = (os.path.join(results_path, "results.csv"))
        
    classes = []
    if network.input_shape == (None, 32, 32, 3):
        classes = classes_cif10
    elif network.input_shape == (None, 28, 28, 1):
        classes = classes_mnist

    with open(csv_file, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        i = 0
        for image in tqdm(dataset, desc="BIM on dataset..."):
            perturbation = 1/255
            adv_generated = False
            fine = False
            misprediction_h5 = False
            misprediction_tflite = False
            misprediction_axc = False
            iterations = 1
            iterations_h5 = 0
            iterations_tflite = 0
            iterations_axc = 0
            perturbation_h5 = 0
            perturbation_tflite = 0
            perturbation_axc = 0
            h5_data = []
            tflite_data = []
            axc_data = []
            data = []
            with tqdm(total = 50, desc = "BIM perturbation on image", leave = False) as perturbations_bar:
                while perturbation < 50/255 and not adv_generated:
                    with tqdm(total = 20, desc = "BIM iteration on image", leave = False) as iteration_bar:
                        while iterations < 20 and not fine:
                            adv_image = iterative_fgsm(network, np.expand_dims(image['image'], axis = 0), image['label'], perturbation, iterations)

                            if not misprediction_h5:
                                adv_h5_prediction = general_predict(network, adv_image, model_type = 0)
                                if np.argmax(image['label']) != np.argmax(adv_h5_prediction):
                                    perturbation_h5 = perturbation
                                    iterations_h5 = iterations
                                    misprediction_h5 = True
                                    adv_image_h5 = adv_image
                                    h5_data = [image['index'], classes[np.argmax(image['label'])], max(image['h5_prediction']),  perturbation_h5, iterations_h5, max(adv_h5_prediction), classes[np.argmax(adv_h5_prediction)]]
                                    h5_adv_vector[image['index']] = adv_image_h5

                            if not misprediction_tflite:
                                adv_tflite_prediction = general_predict(quantized_network, adv_image, model_type = 1)
                                if np.argmax(image['label']) != np.argmax(adv_tflite_prediction):
                                    perturbation_tflite = perturbation
                                    iterations_tflite = iterations
                                    misprediction_tflite = True
                                    adv_image_tflite = adv_image
                                    tflite_data = [max(image['tflite_prediction']), perturbation_tflite, iterations_tflite, max(adv_tflite_prediction), classes[np.argmax(adv_tflite_prediction)]]
                                    tflite_adv_vector[image['index']] = adv_image_tflite

                            if not misprediction_axc:
                                adv_axc_prediction = general_predict(axc_model, adv_image, model_type = 2)
                                if np.argmax(image['label']) != np.argmax(adv_axc_prediction):
                                    perturbation_axc = perturbation
                                    iterations_axc = iterations
                                    misprediction_axc = True
                                    adv_image_axc = adv_image
                                    axc_data = [ max(image['axc_prediction']), perturbation_axc, iterations_axc, max(adv_axc_prediction), classes[np.argmax(adv_axc_prediction)]]
                                    axc_adv_vector[image['index']] = adv_image_axc
                                                                                            
                            if misprediction_tflite and misprediction_h5 and misprediction_axc:
                                fine = True
                                adv_generated = True
                                data = np.concatenate((h5_data, tflite_data, axc_data))
                                writer.writerow(data)
                            else:
                                iterations += 1
                            iteration_bar.update(1)    
                    iterations = 0
                    perturbation += 1/255
                    perturbations_bar.update(1) 

    np.save(os.path.join(results_path, 'h5_images.npy'      ), h5_adv_vector)
    np.save(os.path.join(results_path, 'tflite_images.npy'  ), tflite_adv_vector)
    np.save(os.path.join(results_path, 'axc_images.npy'     ), axc_adv_vector)


def evaluate_deepfool(model, quantized_network, axc_model, dataset, results_path, h5_adv_vector, tflite_adv_vector, axc_adv_vector):
    csv_file = (os.path.join(results_path, "results.csv"))
        
    classes = []
    if model.input_shape == (None, 32, 32, 3):
        classes = classes_cif10
    elif model.input_shape == (None, 28, 28, 1):
        classes = classes_mnist

    network = KerasClassifier(model = model, clip_values = (0, 1))
    with open(csv_file, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        i = 0
        for image in tqdm(dataset, desc="DeepFool on dataset..."):
            perturbation = 1 / 255
            adv_generated = False
            fine = False
            misprediction_h5 = False
            misprediction_tflite = False
            misprediction_axc = False
            iterations = 1
            iterations_h5 = 0
            iterations_tflite = 0
            iterations_axc = 0
            perturbation_h5 = 0
            perturbation_tflite = 0
            perturbation_axc = 0
            h5_data = []
            tflite_data = []
            axc_data = []
            data = []
            while perturbation < 50/255 and not adv_generated:
                while iterations < 20 and not fine:
                    attack_deepfol = DeepFool(classifier = network, max_iter = iterations, epsilon = perturbation, verbose = False)
                    adv_image = np.squeeze(attack_deepfol.generate(np.expand_dims(image['image'], axis = 0)), axis = 0)

                    if not misprediction_h5:
                        adv_h5_prediction = general_predict(model, adv_image, model_type = 0)
                        if np.argmax(image['label']) != np.argmax(adv_h5_prediction):
                            perturbation_h5 = perturbation
                            iterations_h5 = iterations
                            misprediction_h5 = True
                            adv_image_h5 = adv_image
                            h5_data = [image['index'], classes[np.argmax(image['label'])], max(image['h5_prediction']),  perturbation_h5, iterations_h5, max(adv_h5_prediction), classes[np.argmax(adv_h5_prediction)]]
                            h5_adv_vector[image['index']] = adv_image_h5

                    if not misprediction_tflite:
                        adv_tflite_prediction = general_predict(quantized_network, adv_image, model_type = 1)
                        if np.argmax(image['label']) != np.argmax(adv_tflite_prediction):
                            perturbation_tflite = perturbation
                            iterations_tflite = iterations
                            misprediction_tflite = True
                            adv_image_tflite = adv_image
                            tflite_data = [max(image['tflite_prediction']), perturbation_tflite, iterations_tflite, max(adv_tflite_prediction), classes[np.argmax(adv_tflite_prediction)]]
                            tflite_adv_vector[image['index']] = adv_image_tflite
    
                    if not misprediction_axc:
                        adv_axc_prediction = general_predict(axc_model, adv_image, model_type = 2)
                        if np.argmax(image['label']) != np.argmax(adv_axc_prediction):
                            perturbation_axc = perturbation
                            iterations_axc = iterations
                            misprediction_axc = True
                            adv_image_axc = adv_image
                            axc_data = [ max(image['axc_prediction']), perturbation_axc, iterations_axc, max(adv_axc_prediction), classes[np.argmax(adv_axc_prediction)]]
                            axc_adv_vector[image['index']] = adv_image_axc
                                                                                    
                    if misprediction_tflite and misprediction_h5 and misprediction_axc:
                        fine = True
                        adv_generated = True
                        data = np.concatenate((h5_data, tflite_data, axc_data))
                        writer.writerow(data)
                    else:
                        iterations += 1
                iterations = 1
                perturbation += 1/255

    np.save(os.path.join(results_path, 'h5_images.npy'      ), h5_adv_vector)
    np.save(os.path.join(results_path, 'tflite_images.npy'  ), tflite_adv_vector)
    np.save(os.path.join(results_path, 'axc_images.npy'     ), axc_adv_vector)


def evaluate_PGD(model, quantized_network, axc_model, dataset, results_path, h5_adv_vector, tflite_adv_vector, axc_adv_vector):
    csv_file = (os.path.join(results_path, "results.csv"))
        
    classes = []
    if model.input_shape == (None, 32, 32, 3):
        classes = classes_cif10
    elif model.input_shape == (None, 28, 28, 1):
        classes = classes_mnist

    network = KerasClassifier(model = model, clip_values = (0, 1))
    with open(csv_file, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        i = 0
        for image in tqdm(dataset, desc="PGD on dataset..."):
            perturbation = 1 / 255
            adv_generated = False
            fine = False
            misprediction_h5 = False
            misprediction_tflite = False
            misprediction_axc = False
            iterations = 1
            iterations_h5 = 0
            iterations_tflite = 0
            iterations_axc = 0
            perturbation_h5 = 0
            perturbation_tflite = 0
            perturbation_axc = 0
            h5_data = []
            tflite_data = []
            axc_data = []
            data = []
            while perturbation < 50/255 and not adv_generated:
                while iterations < 20 and not fine:
                    attack_pgd = ProjectedGradientDescent(estimator = network, max_iter = iterations, eps = perturbation, eps_step = 1/255, verbose = False, norm = np.inf, targeted = False)
                    adv_image = np.squeeze(attack_pgd.generate(np.expand_dims(image['image'], axis = 0)), axis = 0)

                    if not misprediction_h5:
                        adv_h5_prediction = general_predict(model, adv_image, model_type = 0)
                        if np.argmax(image['label']) != np.argmax(adv_h5_prediction):
                            perturbation_h5 = perturbation
                            iterations_h5 = iterations
                            misprediction_h5 = True
                            adv_image_h5 = adv_image
                            h5_data = [image['index'], classes[np.argmax(image['label'])], max(image['h5_prediction']),  perturbation_h5, iterations_h5, max(adv_h5_prediction), classes[np.argmax(adv_h5_prediction)]]
                            h5_adv_vector[image['index']] = adv_image_h5

                    if not misprediction_tflite:
                        adv_tflite_prediction = general_predict(quantized_network, adv_image, model_type = 1)
                        if np.argmax(image['label']) != np.argmax(adv_tflite_prediction):
                            perturbation_tflite = perturbation
                            iterations_tflite = iterations
                            misprediction_tflite = True
                            adv_image_tflite = adv_image
                            tflite_data = [max(image['tflite_prediction']), perturbation_tflite, iterations_tflite, max(adv_tflite_prediction), classes[np.argmax(adv_tflite_prediction)]]
                            tflite_adv_vector[image['index']] = adv_image_tflite
    
                    if not misprediction_axc:
                        adv_axc_prediction = general_predict(axc_model, adv_image, model_type = 2)
                        if np.argmax(image['label']) != np.argmax(adv_axc_prediction):
                            perturbation_axc = perturbation
                            iterations_axc = iterations
                            misprediction_axc = True
                            adv_image_axc = adv_image
                            axc_data = [ max(image['axc_prediction']), perturbation_axc, iterations_axc, max(adv_axc_prediction), classes[np.argmax(adv_axc_prediction)]]
                            axc_adv_vector[image['index']] = adv_image_axc
                                                                                    
                    if misprediction_tflite and misprediction_h5 and misprediction_axc:
                        fine = True
                        adv_generated = True
                        data = np.concatenate((h5_data, tflite_data, axc_data))
                        writer.writerow(data)
                    else:
                        iterations += 1
                iterations = 1
                perturbation += 1/255

    np.save(os.path.join(results_path, 'h5_images.npy'      ), h5_adv_vector)
    np.save(os.path.join(results_path, 'tflite_images.npy'  ), tflite_adv_vector)
    np.save(os.path.join(results_path, 'axc_images.npy'     ), axc_adv_vector)

def evaluate_CW(model, quantized_network, axc_model, dataset, results_path, h5_adv_vector, tflite_adv_vector, axc_adv_vector):
    csv_file = (os.path.join(results_path, "results.csv"))
        
    classes = []
    if model.input_shape == (None, 32, 32, 3):
        classes = classes_cif10
    elif model.input_shape == (None, 28, 28, 1):
        classes = classes_mnist

    network = KerasClassifier(model = model, clip_values = (0, 1))
    with open(csv_file, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        i = 0
        for image in tqdm(dataset, desc="CW on dataset..."):
            perturbation = 1 / 255
            adv_generated = False
            fine = False
            misprediction_h5 = False
            misprediction_tflite = False
            misprediction_axc = False
            iterations = 1
            iterations_h5 = 0
            iterations_tflite = 0
            iterations_axc = 0
            perturbation_h5 = 0
            perturbation_tflite = 0
            perturbation_axc = 0
            h5_data = []
            tflite_data = []
            axc_data = []
            data = []
            with tqdm(total = 50, desc = "CW iteration on image", leave = False) as iteration_bar:
                while iterations < 50 and not fine:
                    attack_cw = CarliniL2Method(classifier = network, max_iter = iterations, initial_const = 0.1, verbose = False)
                    adv_image = np.squeeze(attack_cw.generate(np.expand_dims(image['image'], axis = 0)), axis = 0)
    
                    if not misprediction_h5:
                        adv_h5_prediction = general_predict(model, adv_image, model_type = 0)
                        if np.argmax(image['label']) != np.argmax(adv_h5_prediction):
                            iterations_h5 = iterations
                            misprediction_h5 = True
                            adv_image_h5 = adv_image
                            perturbation_h5 = 0
                            h5_data = [image['index'], classes[np.argmax(image['label'])], max(image['h5_prediction']),  perturbation_h5, iterations_h5, max(adv_h5_prediction), classes[np.argmax(adv_h5_prediction)]]
                            h5_adv_vector[image['index']] = adv_image_h5
    
                    if not misprediction_tflite:
                        adv_tflite_prediction = general_predict(quantized_network, adv_image, model_type = 1)
                        if np.argmax(image['label']) != np.argmax(adv_tflite_prediction):
                            perturbation_tflite = perturbation
                            iterations_tflite = iterations
                            misprediction_tflite = True
                            adv_image_tflite = adv_image
                            tflite_data = [max(image['tflite_prediction']), perturbation_tflite, iterations_tflite, max(adv_tflite_prediction), classes[np.argmax(adv_tflite_prediction)]]
                            tflite_adv_vector[image['index']] = adv_image_tflite
    
                    if not misprediction_axc:
                        adv_axc_prediction = general_predict(axc_model, adv_image, model_type = 2)
                        if np.argmax(image['label']) != np.argmax(adv_axc_prediction):
                            perturbation_axc = perturbation
                            iterations_axc = iterations
                            misprediction_axc = True
                            adv_image_axc = adv_image
                            axc_data = [ max(image['axc_prediction']), perturbation_axc, iterations_axc, max(adv_axc_prediction), classes[np.argmax(adv_axc_prediction)]]
                            axc_adv_vector[image['index']] = adv_image_axc
                                                                                    
                    if misprediction_tflite and misprediction_h5 and misprediction_axc:
                        fine = True
                        sdv_generated = True
                        data = np.concatenate((h5_data, tflite_data, axc_data))
                        writer.writerow(data)
                    else:
                        iterations += 1
                    iteration_bar.update(1)
                iterations = 1
                        

    np.save(os.path.join(results_path, 'h5_images.npy'      ), h5_adv_vector)
    np.save(os.path.join(results_path, 'tflite_images.npy'  ), tflite_adv_vector)
    np.save(os.path.join(results_path, 'axc_images.npy'     ), axc_adv_vector)
    return


def attack_n_pixel(image, label, bounds, model, model_type, n_pixels, popsize, maxiter):
    
    def object_func(pixels):
        pixels = np.reshape(pixels, (n_pixels, -1))
        x_adv = image.copy()
        rounded_pixels = np.round(pixels).astype(int)
        x_adv[rounded_pixels[:, 0], rounded_pixels[:, 1]] = 0
        adv_prediction = general_predict(model, x_adv, model_type)
        
        if np.argmax(adv_prediction) != np.argmax(label):
            return -max(adv_prediction)
        else:
            return 1000
    
    
    def callback(pixels, convergence):
        solutions = object_func(pixels)
        if (solutions < -0.7):
            return True
        
    pixels = np.zeros((n_pixels, 2))
    result = opt.differential_evolution(object_func, bounds = bounds * n_pixels, maxiter = maxiter, popsize = popsize, callback = callback, workers = 1, polish = False)
    pixels = result.x
    pixels = np.reshape(pixels, (n_pixels, -1))
    
    rounded_pixels = np.round(pixels).astype(int)
    adv = image.copy()
    adv[rounded_pixels[:, 0], rounded_pixels[:, 1]] = 0

    return adv, result.fun, result.nit


def cross_n_p_attacks(model, quantized_model, axc_model, dataset, result_path, h5_adv_vector, tflite_adv_vector, axc_adv_vector):
    csv_file = os.path.join(result_path, "results.csv")

    bounds = [(0, model.input_shape[1] - 1), (0, model.input_shape[1] - 1)]
    with open(csv_file, 'a', encoding = 'UTF8') as f:
        writer = csv.writer(f) 
        
        for  image in tqdm(dataset, desc = "One pixel attack.."):
            h5_fooled =      False
            h5_to_tflite  =  False
            h5_to_axc =      False
            tflite_fooled =  False
            tflite_to_axc =  False
            axc_fooled =     False
            axc_to_h5_back = False
            h5_wprediction = 0
            h5_to_tflite_wprediction = 0
            h5_to_axc_wprediction = 0
            tflite_wprediction = 0
            tflite_to_axc_wprediction = 0
            axc_wprediction = 0
            # popsize != size of the population; population = popsize * n_bounds = popsize * 2 * n_pixels
            adv_image_h5    , result_h5,     nit_h5     = attack_n_pixel(image['image'], image['label'], bounds, model,           n_pixels = 10, popsize = 3, maxiter = 51, model_type = 0)
            adv_image_tflite, result_tflite, nit_tflite = attack_n_pixel(image['image'], image['label'], bounds, quantized_model, n_pixels = 10, popsize = 3, maxiter = 51, model_type = 1)
            adv_image_axc   , result_axc,    nit_axc    = attack_n_pixel(image['image'], image['label'], bounds, axc_model,       n_pixels = 10, popsize = 3, maxiter = 51, model_type = 2)
            h5_adv_prediction =      general_predict(model,             adv_image_h5,       model_type = 0)
            tflite_adv_prediction =  general_predict(quantized_model,   adv_image_tflite,   model_type = 1) 
            axc_adv_prediction =     general_predict(axc_model,         adv_image_axc,      model_type = 2)

            # h5 transfereability to tflite and axc
            if result_h5 < 0 and np.argmax(h5_adv_prediction) != np.argmax(image['label']):
                h5_wprediction = np.max(h5_adv_prediction)
                h5_fooled = True
                h5_adv_vector[image['index']] = adv_image_h5
                h5_to_tflite_predict = general_predict(quantized_model, adv_image_h5, model_type = 1)
                if np.argmax(h5_to_tflite_predict) != np.argmax(image['label']):
                    h5_to_tflite = True
                    h5_to_tflite_wprediction = np.max(h5_to_tflite_predict)
                h5_to_axc_predict = general_predict(axc_model, adv_image_h5, model_type = 2)
                if np.argmax(h5_to_axc_predict) != np.argmax(image['label']):
                    h5_to_axc = True
                    h5_to_axc_wprediction = np.max(h5_to_axc_predict)

            # tflite transfereability to axc
            if result_tflite < 0 and np.argmax(tflite_adv_prediction) != np.argmax(image['label']):
                tflite_fooled = True
                tflite_wprediction = np.max(tflite_adv_prediction)
                tflite_adv_vector[image['index']] = adv_image_tflite
                tflite_to_axc_predict = general_predict(axc_model, adv_image_tflite, model_type = 2)
                if np.argmax(tflite_to_axc_predict) != np.argmax(image['label']):
                    tflite_to_axc = True
                    tflite_to_axc_wprediction = np.max(tflite_to_axc_predict)
                    

            # axc back-transfereability to h5  
            if result_axc < 0 and np.argmax(axc_adv_prediction) != np.argmax(image['label']):
                axc_fooled = True
                axc_wprediction = np.max(axc_adv_prediction)
                axc_adv_vector[image['index']] = adv_image_axc
                axc_to_h5_predict = general_predict(model, adv_image_axc, model_type = 0)
                if np.argmax(axc_to_h5_predict) != np.argmax(image['label']):
                    axc_to_h5_back = True

            data = [image['index'], h5_fooled, nit_h5, h5_wprediction, h5_to_tflite, h5_to_tflite_wprediction, h5_to_axc, h5_to_axc_wprediction, tflite_fooled, nit_tflite, tflite_wprediction, tflite_to_axc, tflite_to_axc_wprediction, axc_fooled, nit_axc, axc_wprediction, axc_to_h5_back]
            writer.writerow(data)
    np.save(os.path.join(result_path, 'tflite_images.npy') , tflite_adv_vector)
    np.save(os.path.join(result_path, 'h5_images.npy'    ) , h5_adv_vector)
    np.save(os.path.join(result_path, 'axc_images.npy') , axc_adv_vector)

    return

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",               "-m",  type = str, help = "Model to attack. Please specify also it's relative path.",                    required = True)
    parser.add_argument("--quantized_model",     "-q",  type = str, help = "Quantized Model to attack. Please specify also it's relative path.",          required = True)
    parser.add_argument("--repo_multipliers",    "-x",  type = str, help = "Path of the repo which containing the approximated multipliers.",             required = True)
    parser.add_argument("--path_muls_conf_file", "-p",  type = str, help = "File that contains axc multipliers details",                                  required = True)
    parser.add_argument("--id_repo",             "-r",  type = str, help = "Repo which will contain results.",                                            required = True)
    parser.add_argument("--min_index",           "-d",  type = int, help = "Minimun index of test dataset",                                               required = True)
    parser.add_argument("--max_index",           "-e",  type = int, help = "Maximum index of test dataset",                                               required = True)
    parser.add_argument("--attacker_knowledge",  "-a",  type = int, help = "If 0, white-box attaks are performed. If 1, black-box attacks are performed", required = False)
    # parser.add_argument("--randomness",         "-n",                        type = int, help = "Number of image to considerate.",                                             required = False)#, default = 500)
    args = parser.parse_args()

    limit_resource_usage()
    
    if "PGD" in args.id_repo or "DeepFool" in args.id_repo or "CW" in args.id_repo:
        tf.compat.v1.disable_eager_execution()

    model_name =  args.model.split("/")[-1].split(".")[0]
    model = tf.keras.models.load_model(args.model)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    # model.summary()
    model_quantized = tf.lite.Interpreter(model_path = args.quantized_model)
    model_quantized.allocate_tensors()
    axc_model = approximate_model(args.quantized_model, model_name, args.repo_multipliers, args.path_muls_conf_file)
    
    _, _, _, _, x_test, y_test, _ = load_dataset(model.input_shape, model_name)
     
    images_indices = [i for i in range(args.min_index, args.max_index)]
    
   # h5_accuracy, tflite_accuracy, axc_accuracy = evaluate_nets(model, model_quantized, axc_model, x_test[images_indices], y_test[images_indices])
   # print(f'H5 accuracy: {h5_accuracy*100}, Tflite accuracy: {tflite_accuracy*100}, AxC accuracy: {axc_accuracy*100}')

    correct_predictions = return_predicted(model, model_quantized, axc_model, x_test[images_indices], y_test[images_indices], images_indices)


    vector_shape = (len(x_test), x_test.shape[1], x_test.shape[2], x_test.shape[3])


    if args.attacker_knowledge == 0:
        results_path_whitebox = os.path.join("..", "WhiteBoxResults", model_name + "_Results" + args.id_repo)
        h5_adv_vector_ifgsm =           np.load(os.path.join(results_path_whitebox, "h5_images.npy"))
        tflite_adv_vector_ifgsm =       np.load(os.path.join(results_path_whitebox, "tflite_images.npy"))
        axc_adv_vector_ifgsm =          np.load(os.path.join(results_path_whitebox, "axc_images.npy"))
    
        if "BIM" in args.id_repo:
            evaluate_speedly(model, model_quantized, axc_model, correct_predictions, results_path_whitebox, h5_adv_vector_ifgsm, tflite_adv_vector_ifgsm, axc_adv_vector_ifgsm)
        elif "PGD" in args.id_repo:
            evaluate_PGD(model, model_quantized, axc_model, correct_predictions, results_path_whitebox, h5_adv_vector_ifgsm, tflite_adv_vector_ifgsm, axc_adv_vector_ifgsm)
        elif "DeepFool" in args.id_repo:
            evaluate_deepfool(model, model_quantized, axc_model, correct_predictions, results_path_whitebox, h5_adv_vector_ifgsm, tflite_adv_vector_ifgsm, axc_adv_vector_ifgsm)
        elif "CW" in args.id_repo:
            evaluate_CW(model, model_quantized, axc_model, correct_predictions, results_path_whitebox, h5_adv_vector_ifgsm, tflite_adv_vector_ifgsm, axc_adv_vector_ifgsm)
        else:
            print("Attack not known")
            exit()
            
    elif args.attacker_knowledge == 1:
        results_path_blackbox = os.path.join("..", "BlackBoxResults", model_name + "_Results" + args.id_repo)
        h5_adv_vector_pix =           np.load(os.path.join(results_path_blackbox, "h5_images.npy"))
        tflite_adv_vector_pix =       np.load(os.path.join(results_path_blackbox, "tflite_images.npy"))
        axc_adv_vector_pix =          np.load(os.path.join(results_path_blackbox, "axc_images.npy"))
        cross_n_p_attacks(model, model_quantized, axc_model, correct_predictions, results_path_blackbox, h5_adv_vector_pix, tflite_adv_vector_pix, axc_adv_vector_pix)

    else:
        print("Attacker knowledge not known")
        exit()
