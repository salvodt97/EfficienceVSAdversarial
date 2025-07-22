import numpy as np
import math
from typing import Literal

def general_predict(model, image, model_type : Literal[0, 1, 2]):
    # Attacco generato sulla rete normale
    if model_type == 0:
        prediction = model.predict(np.expand_dims(image, axis=0), verbose = 0)
        return prediction[0]
    # Attacco generato sulla rete quantizzata
    elif model_type == 1:
        prediction = predict_quantized_model(model, image)
        return prediction
    # Attacco generato sulla rete approssimata
    elif model_type == 2:
        prediction = predict_axc(model, image)
        return prediction
    
def predict_axc(axc_model_struct, image):
    axc_model = axc_model_struct["axc_model"]
    add_par =   axc_model_struct["add_par"]
    mul_par =   axc_model_struct["mul_par"]
    prediction = axc_model.net.predict(image)[0]
    pmin = np.float128(np.min(prediction))
    pmax = np.float128(np.max(prediction))
    for i in range(len(prediction)):
        prediction[i] = np.float128((prediction[i] - pmin)/(pmax - pmin))

    ts = np.float128(0)
    for i in range(len(prediction)):
        prediction[i] = (prediction[i] + add_par)/mul_par
        ts += np.float128(math.exp(prediction[i]))

    for i in range(len(prediction)):
        prediction[i] = np.float128(math.exp(prediction[i])/ts)
    return prediction

def predict_quantized_model(interpreter, image):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    test_image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)
    prediction = output[0]
    return prediction