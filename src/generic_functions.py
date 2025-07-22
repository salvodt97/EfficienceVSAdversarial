import tensorflow as tf, os, numpy as np, csv, shutil, json, json5
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import inspectnn
from inspectnn.Model.GenericModel_tflite import GenericModelTflite

from predict_general import general_predict

classes_cif10 = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
classes_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def resnet50_preprocess_images(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims


def write_wbox_file(results_path):
    header = ['Image', 'Class', 'Correct predtiction from h5', 'Perturbation for h5', 'Number of iterations for h5',
        'Probability of wrong prediction h5', 'Wrong class by h5', 'Correct prediction from tflite', 'Perturbation for tflite',
        'Number of iterations for tflite', 'Probability of wrong prediction tflite', 'Wrong class by tflite', 'Correct prediction from axc', 'Perturbation for axc',
        'Number of iterations for axc', 'Probability of wrong prediction axc', 'Wrong class by axc']
    csv_file = os.path.join(results_path, "results.csv")

    with open(csv_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        
def write_bbox_file(results_path):
    header = ['Image', 'h5_fooled', 'nit_h5', 'h5_wprediction', 'h5_to_tflite', 'h5_to_tflite_wprediction', 'h5_to_axc', 'h5_to_axc_wprediction', 'tflite_fooled', 'nit_tflite', 'tflite_wprediction', 'tflite_to_axc', 'tflite_to_axc_wprediction', 'axc_fooled', 'nit_axc', 'axc_wprediction', 'axc_to_h5_back']  
    csv_file = os.path.join(results_path, "results.csv")

    with open(csv_file, 'w', encoding = 'UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def create_repos(repo):
    # for repo in args:
    if os.path.exists(repo):
        shutil.rmtree(repo)
    if not os.path.exists(repo):
        os.mkdir(repo)

def load_dataset(input_shape, model_name):
    if input_shape == (None, 32, 32, 3):
        # Caricamento del dataset CIFAR-10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
    elif input_shape == (None, 28, 28, 1):
        # Caricamento del dataset MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

    # Preprocess the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    if "50" in model_name:
        x_train = resnet50_preprocess_images(x_train)
        x_test = resnet50_preprocess_images(x_test)
        x_val = resnet50_preprocess_images(x_val)
    else: 
        x_train /= 255
        x_test /= 255
        x_val /= 255

    # Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

    return x_train, y_train, x_val, y_val, x_test, y_test, datagen


def evaluate_nets(model, quant_model, axc_model, dataset, labels):
    h5_accuracy = 0
    tflite_accuracy = 0
    axc_accuracy = 0
    for image, label in tqdm(zip(dataset, labels), total = len(dataset), desc="Eavluating corrected prediction..."):
        prediction = general_predict(model, image, 0)
        if np.argmax(prediction) == np.argmax(label):
            h5_accuracy += 1
        quantized_prediction = general_predict(quant_model, image, 1)
        if np.argmax(quantized_prediction) == np.argmax(label):
            tflite_accuracy += 1
        approximated_prediction = general_predict(axc_model, image, 2)
        if (np.argmax(approximated_prediction) == np.argmax(label)):
            axc_accuracy += 1
    return h5_accuracy/len(dataset), tflite_accuracy/len(dataset), axc_accuracy/len(dataset)       
   


def return_predicted(model, quant_model, axc_model, dataset, labels, original_indices):
    correct_predictions = []
    for index, image, label in tqdm(zip(original_indices, dataset, labels), total = len(dataset), desc="Eavluating corrected prediction..."):
        image_prediction = {}
        prediction = general_predict(model, image, 0)
        quantized_prediction = general_predict(quant_model, image, 1)
        approximated_prediction = general_predict(axc_model, image, 2)
        if np.argmax(prediction) == np.argmax(label) and np.argmax(quantized_prediction) == np.argmax(label) and np.argmax(approximated_prediction) == np.argmax(label):     
            image_prediction['image'] = image
            image_prediction['index'] = index
            image_prediction['label'] = label
            image_prediction['h5_prediction'] = prediction
            image_prediction['tflite_prediction'] = quantized_prediction
            image_prediction['axc_prediction'] = approximated_prediction
            correct_predictions.append(image_prediction)
    return correct_predictions

def limit_resource_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return

def take_quantization_parameters(model_name, file):
    with open(file, 'r') as f:
        json_file = json.load(f)

    add_par = json_file[model_name]['add_parameter']
    mul_par = json_file[model_name]['mul_parameter']
    list_mux = json_file[model_name]['list_multipliers']
    return add_par, mul_par, list_mux


def approximate_model(quantized_model, model_name, path_multipliers, muls_conf_file, file = "../../parameters.json"):
    configuration = json5.load(open(muls_conf_file))
    
    add_par, mul_par, list_mux = take_quantization_parameters(model_name, file)
    axc_model = GenericModelTflite(quantized_model, False)
    axc_model.load_all_multiply(path_multipliers)
    if len(list_mux) > 1:
        axc_model.net.update_multipler(axc_model.generate_multipler_list(list_mux))
    else:
        axc_model.net.update_multipler([axc_model.all_multiplier[list_mux]])
        
    mults_per_layer = np.array([ i.n_moltiplicazioni for i in axc_model.net.layers if isinstance(i, (inspectnn.Conv.ConvLayer.ConvLayer, inspectnn.Dense.DenseLayer.DenseLayer))])
    power_per_layer = [mul["power"] for net_mul in axc_model.generate_multipler_list(list_mux) for mul in configuration["multipliers"] if mul["path"].split("/")[-1].split(".")[0] in str(net_mul).split(".")[-1].split(" ")[0]]
    area_per_layer = [mul["area"] for net_mul in axc_model.generate_multipler_list(list_mux) for mul in configuration["multipliers"] if mul["path"].split("/")[-1].split(".")[0] in str(net_mul).split(".")[-1].split(" ")[0]]  
    baseline_power = np.dot([configuration["multipliers"][0]["power"]] * len(axc_model.generate_multipler_list(list_mux)), mults_per_layer) / 65536   
    baseline_area = np.sum([configuration["multipliers"][0]["area"]] * len(axc_model.generate_multipler_list(list_mux)))
    approx_power = np.dot(power_per_layer, mults_per_layer) / 65536
    approx_area = np.sum(area_per_layer)
    floating_point_power = 2.048 * np.sum(mults_per_layer) / 65536
    floating_point_area = 1215.088 * len(axc_model.generate_multipler_list(list_mux))
    
    axc_model_struct = {}
    axc_model_struct["axc_model"]    = axc_model
    axc_model_struct["add_par"]      = add_par
    axc_model_struct["mul_par"]      = mul_par
    axc_model_struct["baseline_power"]   = baseline_power
    axc_model_struct["baseline_area"]    = baseline_area
    axc_model_struct["axc_power"]   = approx_power
    axc_model_struct["axc_area"]    = approx_area
    axc_model_struct["floating_point_area"]    = floating_point_area
    axc_model_struct["floating_point_power"]    = floating_point_power
    return axc_model_struct
