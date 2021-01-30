import sys
import os
import PySimpleGUI as sg


def WelcomeWindow():
    r"""Embodies the Welcome Frame.
    """
    # Set layout
    sg.theme("SystemDefault")
    layout = [[sg.Text("Hello and Welcome to our System for Surgical Tool Recognition :)", font=("Helvetica", 18))],
              [sg.Image(os.path.join(os.getcwd(), 'cai', 'gui', 'icon.png'), pad = (0, 10))],
              [sg.Button("Continue", pad=(30, 10)), sg.Button("Exit", pad=(30, 0))]]
              
    sg.set_options(text_justification="center")

    # Create the window
    window = sg.Window("Surgical Tool Recognition", layout, font=("Helvetica", 14), element_justification = "center")
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Exit":
            return False
        if event == "Continue":
            window.close()
            return True


def StartWindow(input):
    r"""Embodies the Start Frame where the user specifies a video path to the video
    for which he wants to predict the surgery tools in it.
    """
    # Set layout
    sg.theme("SystemDefault")
    layout = [[sg.Text("Paste/Browse the path to the video for which you want the surgical tools to be predicted:", font=("Helvetica", 14))],
              [sg.Text("Choose a video: "), sg.Input(input), sg.FileBrowse(key="-IN-")],
              [sg.Button("Submit"), sg.Button("Cancel")]]
              
    sg.set_options(text_justification="left")
    
    # Create the window
    window = sg.Window("Surgical Tool Recognition", layout, font=("Helvetica", 14))
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
              [sg.ProgressBar(2, orientation="h",
                              size=(20, 20), key="progress")],
              [sg.Cancel()]]
    # Create the window
    window = sg.Window("Loading your video", layout, font=("Helvetica", 14))
    progress_bar = window["progress"]
    
    return window, progress_bar
    
    
def TransformVideo(input, video_info):
    r"""Embodies the Frame that shows extracted information about the video during the load
    process and indicates that the video needs to be transformed for the model, if it is not
    already. The user can specify if he wants to save the video as a specified target path or
    not.
    param: video_info, a dict with video information needed for the window
    """
    video_path = [[sg.Text("video at: ", size=(12, 1)), sg.Text(str(video_info["path"]))]]
    video_specs = [[sg.Text("nr_frames: ", size=(12, 1)), sg.Text(str(video_info["frames"]), size=(6, 1))],
                   [sg.Text("frame_size: ", size=(12, 1)), sg.Text(str(video_info["size"]), size=(15, 1))],
                   [sg.Text("fps: ", size=(12, 1)), sg.Text(str(video_info["fps"]), size=(6, 1))],
                   [sg.Text("video_length: ", size=(12, 1)), sg.Text(str(video_info["length"]), size=(15, 1))]]
    trans = [[sg.Text("Your input video needs to be transformed to the models input size.")]]

    layout = [[sg.Frame("Your chosen video", video_path, title_color="black", font=("Helvetica", 14))],
              [sg.Frame("Video specifications", video_specs, title_color="black", font=("Helvetica", 14))]]
              
    # Check if video needs to be transformed
    if int(video_info["fps"]) != 1 or str(video_info["size"]) != "(224, 224, 3)":
        layout.append([sg.Frame("Transformation needed", trans, title_color="black", font=("Helvetica", 14))])
        layout.append([sg.Text("Choose a target path: "), sg.Input(input), sg.FolderBrowse(key="-IN-")])
        layout.append([sg.Button("Back"), sg.Button("Continue"), sg.Button("Cancel")])
    else:
        layout.append([sg.Text("Choose a target path: "), sg.Input(input), sg.FolderBrowse(key="-IN-")])
        layout.append([sg.Button("Back"), sg.Button("Continue"), sg.Button("Cancel")])

    sg.set_options(text_justification="left")

    window = sg.Window("Pre-trained Model Parameters", layout, font=("Helvetica", 14))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False, None, False
        if event == "Back":
            window.close()
            return True, values[0], True
        if not os.path.isdir(values[0]):
            sg.popup_error("Your folder path", values[0],
                           "is a dead end --> Directory not found.", font=("Helvetica", 14))
            continue
        if event == "Continue":
            window.close()
            return True, values[0], False
    
    
def TransformVideoProgress():
    r"""Embodies the Frame that shows the progress during transformation of the video with moviepy.
    Loop needs to be modified during the loading and extracting specifications of the video.
    --> Needs to be done by calling function.
    """
    # Set layout
    layout = [[sg.Text("Your video is being transformed", font=("Helvetica", 14))],
              [sg.ProgressBar(3, orientation="h",
                              size=(20, 20), key="progress")],
              [sg.Cancel()]]
    # Create the window
    window = sg.Window("Transforming your video", layout, font=("Helvetica", 14))
    progress_bar = window["progress"]
    
    return window, progress_bar
    
    
def ChooseModelAndDevice(selected_model, model_names, cpu, gpu):
    r"""This window shows the models a user can choose for prediction and a field
    where the device needs to be specified: CPU or GPU.
    param: model_names, a tuple with the possible model names, e.g. ("CNN", "ResNet")
    """
    choose = [[sg.Text("Choose a pre-trained model: ", size=(24, 1)), sg.Combo(values=model_names, key="model", default_value = selected_model)],
              [sg.Text("Choose a device for prediction: ", size=(24, 1)),
               sg.Radio("CPU", 1, key="cpu", default=cpu),
               sg.Radio("GPU", 1, key="gpu", default=gpu)],
              [sg.Spin([i for i in range(0,8)], initial_value=0, key="gpu_id"), sg.Text("GPU ID")]]
    
    layout = [[sg.Frame("Your preferences are needed", choose, title_color="black", font=("Helvetica", 14))],
              [sg.Button("Back"), sg.Button("Continue"), sg.Button("Cancel")]]

    sg.set_options(text_justification="left")
    
    window = sg.Window("Prediction preferences", layout, font=("Helvetica", 14))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False, None, False
        if values["gpu"]:
            result = [values["model"], values["gpu_id"]]
        else:
            result = [values["model"], None]
        if event == "Back":
            window.close()
            return True, result, True
        if event == "Continue":
            window.close()
            return True, result, False
    
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
              [sg.Button("Back"), sg.Button("Continue"), sg.Button("Cancel")]]

    sg.set_options(text_justification="left")

    window = sg.Window("Pre-trained Model Parameters", layout, font=("Helvetica", 14))
    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED or event == "Cancel":
            return False, False
        if event == "Back":
            window.close()
            return True, True
        elif event == "Continue":
            window.close()
            return True, False
            
            
def PredictVideoTools(frames):
    r"""Embodies the Frame that shows the progress during predicting the video tools with
    The pre-trained model. Loop needs to be modified during the prediction.
    --> Needs to be done by calling function.
    """
    # Set layout
    layout = [[sg.Text("The video tools are being predicted", font=("Helvetica", 14))],
              [sg.ProgressBar(frames, orientation="h",
                              size=(20, 20), key="progress")],
              [sg.Cancel()]]
    # Create the window
    window = sg.Window("Predict used tools", layout, font=("Helvetica", 14))
    progress_bar = window["progress"]
    
    return window, progress_bar
    

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
        if event == sg.WIN_CLOSED or event == "Finish":
            return False
        elif event == "Start again":
            window.close()
            return True
   

def ErrorWindow(name, msg):
    r"""This window prints the message into an error window.
    """
    # Set layout
    layout = [[sg.Text("The following error occured: \n")],
              [sg.Output(key='-OUT-', size=(45, 10))],
              [sg.Button("Start again"), sg.Button("Exit")]]
                  
    # Create the window
    window = sg.Window(name, layout, font=("Helvetica", 14)).finalize()
    window['-OUT-'].TKOut.output.config(wrap='word')
    print(msg)
        
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            return False
        elif event == "Start again":
            window.close()
            return True
