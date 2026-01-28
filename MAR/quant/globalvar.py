# catch diffusion input for calibration
global diffusion_input_list
diffusion_input_list = []

def appendInput(value):
    diffusion_input_list.append(value)

def getInputList():
    return diffusion_input_list
    
def removeInput():
    diffusion_input_list.clear()

global diffusion_output_list
diffusion_output_list = []

def appendOutput(value):
    diffusion_output_list.append(value)

def getOutputList():
    return diffusion_output_list
    
def removeOutput():
    diffusion_output_list.clear()
