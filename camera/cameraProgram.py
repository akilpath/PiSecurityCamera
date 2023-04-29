import tensorflow as tf

MODELPATH = ""
interpreter = tf.lite.Interpreter(model_path = MODELPATH)
interpreter.allocate_tensors()

inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

inputShape = inputDetails[0]["shape"]


