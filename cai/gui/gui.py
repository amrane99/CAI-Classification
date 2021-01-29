import sys
import os
import PySimpleGUI as sg

def WelcomeWindow():
    r"""Embodies the Welcome Frame where the user specifies a video path to the video
    for which he wants to predict the surgery tools in it.
    """
    # Set layout
    sg.theme("SystemDefault")
    layout = [[sg.Text("Hello and Welcome to our System for Surgical Tool Recognition :)", font=("Helvetica", 16))],
         [sg.Text("Paste/Browse the path to the video for which you want the surgical tools to be predicted:", font=("Helvetica", 14))],
         [sg.Text("Choose a video: "), sg.Input(), sg.FileBrowse(key="-IN-")],
         [sg.Button("Submit"), sg.Button("Cancel")]]
    # Create the window
    window = sg.Window("Surgical Tool Recognition", layout, font=("Helvetica", 14), size=(600,150))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False, None
        if event == "Submit":
            if "." not in values[0] or values[0].split(".")[1] != "mp4":
                sg.popup_error("Your Input", values["-IN-"],
                               "is no valid video path (needs to end with .mp4).", font=("Helvetica", 14))
                continue
            if not os.path.isfile(values[0]):
                sg.popup_error("Your video path", values["-IN-"],
                               "is a dead end --> File not found.", font=("Helvetica", 14))
                continue
            else:
                sg.popup_ok("Your Input video:", values["-IN-"], font=("Helvetica", 14))
            window.close()
            return True, values[0]


def LoadVideo():
    r"""Embodies the Frame that shows the progress during loading the video with moviepy.
    Loop needs to be modified during the loading and extracting specifications of the video.
    --> Needs to be done by calling function.
    """
    # Set layout
    layout = [[sg.Text("Your video is being loaded", font=("Helvetica", 14))],
              [sg.ProgressBar(3, orientation="h",
                              size=(20, 20), key="progress")],
              [sg.Cancel()]]
    # Create the window
    window = sg.Window("Extracting your video based on path", layout)
    progress_bar = window["progress"]
    
    # return window
    
    # Loop that shows progress bar
    for i in range(3):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key="timeout")
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False
        # Update bar with loop value +1
        progress_bar.update_bar(i+1)
    window.CloseNonBlocking()
    return True
    
    
def TransformVideo(video_info):
    r"""Embodies the Frame that shows extracted information about the video during the load
    process and indicates that the video needs to be transformed for the model, if it is not
    already. The user can specify if he wants to save the video as a specified target path or
    not.
    param: video_info, a dict with video information needed for the window
    """
    video_path = [[sg.Text("video at: ", size=(12, 1)), sg.Text(str(video_info["path"]))]]
    video_specs = [[sg.Text("nr_frames: ", size=(12, 1)), sg.Text(str(video_info["frames"]), size=(6, 1))],
                   [sg.Text("frame_size: ", size=(12, 1)), sg.Text(str(video_info["size"]), size=(10, 1))],
                   [sg.Text("fps: ", size=(12, 1)), sg.Text(str(video_info["fps"]), size=(6, 1))],
                   [sg.Text("video_length: ", size=(12, 1)), sg.Text(str(video_info["length"]), size=(15, 1))]]
    trans = [[sg.Text("Your input video needs to be transformed to the models input size.")]]

    layout = [[sg.Frame("Your chosen video", video_path, title_color="black", font=("Helvetica", 14))],
              [sg.Frame("Video specifications", video_specs, title_color="black", font=("Helvetica", 14))]]
              
    # Check if video needs to be transformed
    if int(video_info["fps"]) != 1 or str(video_info["size"]) != "(224, 224, 3)":
        layout.append([sg.Frame("Transformation needed", trans, title_color="black", font=("Helvetica", 14))])
        layout.append([sg.Radio("Yes", 1, key="yes", default=True),
                       sg.Radio("No", 1, key="no")])
        layout.append([sg.Text("Choose a target path: "), sg.Input(), sg.FolderBrowse(key="-IN-")])
        layout.append([sg.Button("Continue"), sg.Button("Cancel")])
    else:
        layout.append([sg.Button("Continue"), sg.Button("Cancel")])

    sg.set_options(text_justification="left")

    window = sg.Window("Pre-trained Model Parameters", layout, font=("Helvetica", 14))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False, None
        if values["yes"]:
            if not os.path.isdir(values[0]):
                sg.popup_error("Your folder path", values[0],
                               "is a dead end --> Directory not found.", font=("Helvetica", 14))
                continue
        if event == "Continue":
            window.close()
            return True, values[0]
    
    
def ChooseModelAndDevice(model_names):
    r"""This window shows the models a user can choose for prediction and a field
    where the device needs to be specified: CPU or GPU.
    param: model_names, a tuple with the possible model names, e.g. ("CNN", "ResNet")
    """
    choose = [[sg.Text("Choose a pre-trained model: ", size=(24, 1)), sg.Combo(values=model_names, key="model", default_value=model_names[0])],
              [sg.Text("Choose a device for prediction: ", size=(24, 1)),
               sg.Radio("CPU", 1, key="cpu", default=True),
               sg.Radio("GPU", 1, key="gpu")],
              [sg.Spin([i for i in range(0,8)], initial_value=0, key="gpu_id"), sg.Text("GPU ID")]]
    
    layout = [[sg.Frame("Your preferences are needed", choose, title_color="black", font=("Helvetica", 14))],
              [sg.Button("Continue"), sg.Button("Cancel")]]

    sg.set_options(text_justification="left")
    
    window = sg.Window("Prediction preferences", layout, font=("Helvetica", 14))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False, None
        if values["gpu"]:
            result = [values["model"], values["gpu_id"]]
        else:
            result = [values["model"], None]
        if event == "Continue":
            window.close()
            return True, result
    
def ModelSpecs(config):
    r"""This window shows the parameters with which the model has been trained.
    param: config, a dict with information needed for the window
    """
    choosen_model = [[sg.Text("model: ", size=(12, 1)), sg.Text(str(config["model"]))]]
    command_line_parms = [[sg.Text("dataset: ", size=(12, 1)), sg.Text("Cholec80")],
                          [sg.Text("number of tools: ", size=(12, 1)), sg.Text("7", size=(6, 1))],
                          [sg.Text("nr_epochs: ", size=(11, 1)), sg.Text("  "+str(config["nr_epochs"]), size=(6, 1)), sg.Text("  batch_size: ", size=(11, 1)), sg.Text("   "+str(config["batch_size"]), size=(6, 1))],
                          [sg.Text("loss: ", size=(12, 1)), sg.Text("Binary Cross Entropy Loss")],
                          [sg.Text("val_ratio: ", size=(12, 1)), sg.Text("0.125", size=(6, 1)), sg.Text("test_ratio: ", size=(12, 1)),
                           sg.Text("0.125", size=(6, 1))],
                          [sg.Text("learning_rate: ", size=(12, 1)), sg.Text(str(config["learning_rate"]), size=(6, 1)), sg.Text("weight_decay: ", size=(12, 1)), sg.Text(str(config["weight_decay"]), size=(8, 1))]]
    video_specs = [[sg.Text("nr_videos: ", size=(12, 1)), sg.Text("80", size=(6, 1))],
                   [sg.Text("nr_frames: ", size=(12, 1)), sg.Text("2000", size=(6, 1))],
                   [sg.Text("frame_size: ", size=(12, 1)), sg.Text("(224, 224, 3)", size=(10, 1))]]
    accuracy = [[sg.Text("Test accuracy of the model: ", size=(24, 1)), sg.Text(str(config["test_acc"])+"%", size=(6, 1))]]

    layout = [[sg.Frame("Your choosen model for the prediction", choosen_model, title_color="black", font=("Helvetica", 14))],
              [sg.Frame("Model parameter settings", command_line_parms, title_color="black", font=("Helvetica", 14))],
              [sg.Frame("Model accuracy", accuracy, title_color="black", font=("Helvetica", 14))],
              [sg.Frame("Video specifications", video_specs, title_color="black", font=("Helvetica", 14))],
              [sg.Button("Continue"), sg.Button("Cancel")]]

    sg.set_options(text_justification="left")

    window = sg.Window("Pre-trained Model Parameters", layout, font=("Helvetica", 14))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False
        elif event == "Continue":
            window.close()
            return True
            
            
def PredictVideoTools():
    r"""Embodies the Frame that shows the progress during predicting the video tools with
    The pre-trained model. Loop needs to be modified during the prediction.
    --> Needs to be done by calling function.
    """
    # Set layout
    layout = [[sg.Text("The video tools are being predicted", font=("Helvetica", 14))],
              [sg.ProgressBar(3, orientation="h",
                              size=(20, 20), key="progress")],
              [sg.Cancel()]]
    # Create the window
    window = sg.Window("Extractin video based on path", layout)
    progress_bar = window["progress"]
    
    # return window
    
    # Loop that shows progress bar
    for i in range(3):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key="timeout")
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False
        # Update bar with loop value +1
        progress_bar.update_bar(i+1)
    window.CloseNonBlocking()
    return True
    

def ResultWindow(result):
    r"""This window prints the detected tools into a listbox.
    """
    # Set layout
    layout = [[sg.Text("These are the tools predicted using the specified model: \n")],
              [sg.Output(key='-OUT-', size=(45, 10))],
              [sg.Button("Start again"), sg.Button("Finish")]]
    # Create the window
    window = sg.Window("Results", layout, font=("Helvetica", 14)).finalize()
    window['-OUT-'].TKOut.output.config(wrap='word')
    print(result)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Finish":
            return False
        elif event == "Start again":
            return True
    
    
if __name__ == "__main__":
    # Define samples
    config = {"nr_epochs": 300, "batch_size": 50, "learning_rate": 0.001, "weight_decay": 0.75, "model": "AlexNet", "test_acc": 80}
    model_names = ("ResNet", "AlexNet", "CNN")
    video_info = {"path": "your path", "frames": 1323, "fps": 25, "length": "1 h - 13 m - 12 s", "size": "(244, 224, 3)"}
    output = "Frame <nr> (video_time) \t --> \t Present Tools\nFrame 1 (0 h - 0 m - 1s)  \t --> \t Grasper, Hook\nFrame 2 (0 h - 0 m - 2s) \t --> \t None"
    
    # Run GUI components
    #if not WelcomeWindow()[0]:
    #    sys.exit()
    if not LoadVideo():
        sys.exit()
    if not TransformVideo(video_info)[0]:
        sys.exit()
    if not ChooseModelAndDevice(model_names)[0]:
        sys.exit()
    if not ModelSpecs(config):
        sys.exit()
    if not PredictVideoTools():
        sys.exit()
    if not ResultWindow(output)[0]:
        sys.exit()

"""
def Output_Window():
    #layout = [[sg.T("")], [sg.Text("Hello and Welcome to our System of Surgical Tool Recognition!\n Choose a file: "), sg.Input(), sg.FileBrowse()]]
    sg.theme("SystemDefault")
    layout = [[sg.T("Thank you for using Surgical Tool Recognition System!")],
         [sg.T("Please, write/browse the path where you want the resukts to be saved:")],
         [sg.Text("Choose a file: "), sg.Input(), sg.FolderBrowse(key="-IN-")],
         [sg.Button("Continue"), sg.Button("Cancel")]]
    # Create the window
    window = sg.Window("Output Path", layout, size=(600,150))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            break
        elif event == "Continue":
            sg.popup_ok_cancel("The results will be saved in ", values["-IN-"],
                                "Do you wish to proceed?")
            break


    window.close()

def CustomMeter2():
    # layout the form
    layout = [[sg.Text("The results are being calculated")],
              [sg.ProgressBar(3, orientation="h",
                              size=(20, 20), key="progress")],
              [sg.Cancel()]]

    # create the form`
    window = sg.Window("Custom Progress Meter", layout)
    progress_bar = window["progress"]
    
    # return window
    
    # loop that would normally do something useful
    for i in range(3):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key="timeout")
        if event == "Cancel" or event == None:
            break
        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.update_bar(i+1)
    # done with loop... need to destroy the window as it"s still open
    window.CloseNonBlocking()



def Result_Window():
    layout = [[sg.Text("These are the results for the Tool Recognition System")],
              [sg.Output(size=(60, 20))],
              [sg.Finish(), sg.Cancel()]]
    window = sg.Window("Results Window", layout)
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event=="Cancel":
            break
        elif event == "Ok":
            sg.popup_ok_cancel("Do you want to use another video?")
            break
"""
