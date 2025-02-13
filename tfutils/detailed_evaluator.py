def load_tflite_model_runtime(model_path):
    import tensorflow as tf

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter

def run_model(interpreter, signs, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    return signs[output_data.argmax()]

if __name__ == "__main__":
    model_path = "/models/250_cnn_2d.tflite"
    signs_list = "/meta/250_sign_list.txt"

    with open(signs_list) as f:
        signs = [i.strip().lower() for i in f.readlines()]
    
    # Load using tflite_runtime
    interpreter_runtime = load_tflite_model_runtime(model_path)
    print("Model loaded with tflite_runtime")