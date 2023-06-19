import numpy
import tensorflow as tf
# import picamera
from PIL import Image
import numpy as np
from time import sleep


# camera = picamera.PiCamera()
# camera.resolution(480,640)
# outputObj = np.empty((480,640,3), dtype=np.uint8)
# camera.capture(outputObj, "rgb")

def piTest():
    MODELPATH = "./model20230415.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODELPATH)
    interpreter.allocate_tensors()

    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()

    running = True
    camera = None
    camera.resolution = (500,500)

    while (running):
        sleep(5)
        outputObj = np.empty((500, 500, 3), dtype=np.float32)
        camera.capture(outputObj, "rgb")
        inputData = outputObj[np.newaxis, :]
        interpreter.set_tensor(inputDetails[0]["index"], inputData)
        interpreter.invoke()

        output_data = interpreter.get_tensor(outputDetails[0]["index"])
        print(output_data)

def computerTest():
    MODELPATH = "./model20230618.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODELPATH)
    interpreter.allocate_tensors()

    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()

    running = True
    samples = []
    for i in range(3):
        sample = np.array(Image.open(f"../sample/sample{i + 1}.jpg"), dtype=np.float32)
        sample = sample[np.newaxis, :]
        samples.append(sample)

    for sample in samples:
        interpreter.set_tensor(inputDetails[0]["index"], sample)
        interpreter.invoke()
        output_data = interpreter.get_tensor(outputDetails[0]["index"])
        print(output_data)


computerTest()
