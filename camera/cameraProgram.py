import tensorflow as tf
import picamera
import numpy as np
from time import sleep

camera = picamera.PiCamera()
camera.resolution(480,640)
outputObj = np.empty((480,640,3), dtype=np.uint8)
camera.capture(outputObj, "rgb")

MODELPATH = "../model20230415.tflite"
interpreter = tf.lite.Interpreter(model_path=MODELPATH)
interpreter.allocate_tensors()

inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

running = True
while(running):
    sleep(5)
    outputObj = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(outputObj, "rgb")
    inputData = outputObj
    interpreter.set_tensor(inputDetails[0]["index"], inputData)
    interpreter.invoke()
    output_data = interpreter.get_tensor(outputDetails[0]["index"])
    print(output_data)




