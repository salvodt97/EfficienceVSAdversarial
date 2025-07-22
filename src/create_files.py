import tensorflow as tf, os, argparse, numpy as np

from generic_functions import load_dataset, create_repos, write_bbox_file, write_wbox_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              "-m",   type = str, help = "Model to attack. Please specify also it's relative path.",                    required = True)
    parser.add_argument("--id_repo",            "-r",   type = str, help = "Repo which will contain results.",                                            required = True)
    parser.add_argument("--attacker_knowledge", "-a",   type = int, help = "If 0, white-box attaks are performed. If 1, black-box attacks are performed", required = True)
    args = parser.parse_args()
    
    model_name =  args.model.split("/")[-1].split(".")[0]
    model = tf.keras.models.load_model(args.model)
    
    #_, _, _, _, x_test, _, _ = load_dataset(model.input_shape)
    
    #vector_shape = (len(x_test), x_test.shape[1], x_test.shape[2], x_test.shape[3])
    vector_shape = (10000, 32, 32, 3)
    modelResults = "results.csv"
    


    if args.attacker_knowledge == 0:
        results_path_whitebox = os.path.join("..", "WhiteBoxResults", model_name + "_Results" + args.id_repo)
        create_repos(results_path_whitebox)
        write_wbox_file(results_path_whitebox)
        np.save(os.path.join(results_path_whitebox, 'h5_images.npy'), np.zeros(vector_shape, dtype = np.float32))
        np.save(os.path.join(results_path_whitebox, 'tflite_images.npy'), np.zeros(vector_shape, dtype = np.float32))
        np.save(os.path.join(results_path_whitebox, 'axc_images.npy'), np.zeros(vector_shape, dtype = np.float32))
    elif args.attacker_knowledge == 1:
        results_path_blackbox = os.path.join("..", "BlackBoxResults", model_name + "_Results" + args.id_repo)
        create_repos(results_path_blackbox)
        write_bbox_file(results_path_blackbox)
        np.save(os.path.join(results_path_blackbox, 'h5_images.npy'), np.zeros(vector_shape, dtype = np.float32))
        np.save(os.path.join(results_path_blackbox, 'tflite_images.npy'), np.zeros(vector_shape, dtype = np.float32))
        np.save(os.path.join(results_path_blackbox, 'axc_images.npy'), np.zeros(vector_shape, dtype = np.float32))
