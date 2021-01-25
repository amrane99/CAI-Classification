import sys
import PySimpleGUI as sg

def MachineLearningGUI():
    sg.set_options(text_justification='right')

    flags = [[sg.CB('Cross-Validation', size=(12, 1), default=False), sg.CB('Resize', size=(20, 1), default=False)],
             [sg.CB('Augmentation', size=(12, 1), default=False), sg.CB(
                 'Random Frames', size=(20, 1), default=True)],
             [sg.CB('Write Results', size=(12, 1)), sg.CB(
                 'Keep Intermediate Data', size=(20, 1))],
             [sg.CB('Normalize', size=(12, 1), default=True),
              sg.CB('Verbose', size=(20, 1))],]
        
    loss_functions = [[sg.Rad('Binary-Cross-Entropy', 'loss', size=(12, 1)), sg.Rad('Logistic', 'loss', default=True, size=(12, 1))],
                      [sg.Rad('Hinge', 'loss', size=(12, 1)),
                       sg.Rad('MSE', 'loss', size=(12, 1))],]

    command_line_parms = [[sg.Text('device', size=(12, 1)), sg.Drop(values=('cuda:5', 'gpu')),
                           sg.Text('nr_runs', size=(12, 1), pad=((7, 3))), sg.Spin(values=[i for i in range(1, 100)], initial_value=1, size=(6, 1))],
                          [sg.Text('val_ratio', size=(12, 1)), sg.Input(default_text='0.125', size=(6, 1)), sg.Text('test_ratio', size=(12, 1)),
                           sg.Input(default_text='0.125', size=(6, 1))],
                          [sg.Text('input_shape', size=(12, 1)), sg.Input(default_text='(3, 224, 224)', size=(10, 1)), sg.Text('learning_rate', size=(12, 1)),
                           sg.Input(default_text='0.001', size=(6, 1))],
                          [sg.Text('batch_size', size=(11, 1), pad=((7, 3))), sg.Spin(values=[i for i in range(1, 1000, 10)], initial_value=50, size=(6, 1)), 
                           sg.Text('number of tools', size=(12, 1)), sg.Spin(values=[i for i in range(1, 10)], initial_value=7, size=(6, 1))], 
                          [sg.Text('nr_epochs', size=(11, 1), pad=((7, 3))), sg.Spin(values=[i for i in range(1, 1000, 10)], initial_value=50, size=(6, 1)), sg.Text('nr_videos', size=(12, 1)),
                           sg.Input(default_text='80', size=(6, 1))],
                          [sg.Text('weight_decay', size=(12, 1)), sg.Input(default_text='0.75', size=(8, 1)), sg.Text('nr_frames', size=(12, 1)),
                           sg.Input(default_text='2000', size=(6, 1))],
                          [sg.Text('save_interval', size=(12, 1)), sg.Input(default_text='25', size=(6, 1)), sg.Text('bot_msg_interval', size=(12, 1)),
                           sg.Input(default_text='5', size=(6, 1))],
                          [sg.Text('dataset', size=(12, 1)), sg.Drop(values=('Cholec80', 'other')),
                           sg.Text('model', size=(12, 1)), sg.Drop(values=('CNN', 'AlexNet', 'ResNet')),]]

    layout = [[sg.Frame('Command Line Parameteres', command_line_parms, title_color='green', font='Any 12')],
              [sg.Frame('Flags', flags, font='Any 12', title_color='blue')],
              [sg.Frame('Loss Functions',  loss_functions,
                        font='Any 12', title_color='red')],
              [sg.Submit(), sg.Cancel()]]

    sg.set_options(text_justification='left')

    window = sg.Window('Machine Learning Front End',
                       layout, font=("Helvetica", 12))
    button, values = window.read()
    window.close()
    print(button, values)


def CustomMeter():
    # layout the form
    layout = [[sg.Text('Your video is being loaded')],
              [sg.ProgressBar(10000, orientation='h',
                              size=(20, 20), key='progress')],
              [sg.Cancel()]]

    # create the form`
    window = sg.Window('Custom Progress Meter', layout)
    progress_bar = window['progress']
    # loop that would normally do something useful
    for i in range(10000):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key='timeout')
        if event == 'Cancel' or event == None:
            break
        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.update_bar(i+1)
    # done with loop... need to destroy the window as it's still open
    window.CloseNonBlocking()


def Window_1st():

	#layout = [[sg.T("")], [sg.Text("Hello and Welcome to our System of Surgical Tool Recognition!\n Choose a file: "), sg.Input(), sg.FileBrowse()]]
	sg.theme("DarkTeal2")
	layout = [[sg.T("Hello and Welcome to our System of Surgical Tool Recognition!")], 
	     [sg.T("Write/Browse the path from the video you want the surgical tools to be recognized:")],
	     [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],
	     [sg.Button("Submit"), sg.Button("Cancel")]]
	# Create the window
	window = sg.Window("Surgical Tool Recognition System", layout, size=(600,150))
	#window2 = sg.Window(['You entered', values["-IN-"]],
	#       			['Do you wish to proceed?'],
	#      			[sg.Button("Yes"), sg.Button("Cancel")])
	# Create an event loop
	while True:
		event, values = window.read()
		# End program if user closes window or
		# presses the OK button
		if event == sg.WIN_CLOSED or event=="Cancel":
		    break
		elif event == "Submit":
		    sg.popup_ok_cancel('You entered', values["-IN-"],
		    					'Do you wish to proceed?')
		    break


	window.close()


def Output_Window():

	#layout = [[sg.T("")], [sg.Text("Hello and Welcome to our System of Surgical Tool Recognition!\n Choose a file: "), sg.Input(), sg.FileBrowse()]]
	sg.theme("DarkTeal2")
	layout = [[sg.T("Thank you for using Surgical Tool Recognition System!")], 
	     [sg.T("Please, write/browse the path where you want the resukts to be saved:")],
	     [sg.Text("Choose a file: "), sg.Input(), sg.FolderBrowse(key="-IN-")],
	     [sg.Button("Submit"), sg.Button("Cancel")]]
	# Create the window
	window = sg.Window("Output Path", layout, size=(600,150))
	#window2 = sg.Window(['You entered', values["-IN-"]],
	#       			['Do you wish to proceed?'],
	#      			[sg.Button("Yes"), sg.Button("Cancel")])
	# Create an event loop
	while True:
		event, values = window.read()
		# End program if user closes window or
		# presses the OK button
		if event == sg.WIN_CLOSED or event=="Cancel":
		    break
		elif event == "Submit":
		    sg.popup_ok_cancel('The results will be saved in ', values["-IN-"],
		    					'Do you wish to proceed?')
		    break


	window.close()

def CustomMeter2():
    # layout the form
    layout = [[sg.Text('The results are being calculated')],
              [sg.ProgressBar(10000, orientation='h',
                              size=(20, 20), key='progress')],
              [sg.Cancel()]]

    # create the form`
    window = sg.Window('Custom Progress Meter', layout)
    progress_bar = window['progress']
    # loop that would normally do something useful
    for i in range(10000):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key='timeout')
        if event == 'Cancel' or event == None:
            break
        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.update_bar(i+1)
    # done with loop... need to destroy the window as it's still open
    window.CloseNonBlocking()



def Result_Window():
	layout = [[sg.Text('These are the results for the Tool Recognition System')],
              [sg.Output(size=(60, 20))],
              [sg.Cancel(), sg.Ok()]]
	window = sg.Window("Results Window", layout)
	while True:
		event, values = window.read()
		# End program if user closes window or
		# presses the OK button
		if event == sg.WIN_CLOSED or event=="Cancel":
		    break
		elif event == "Ok":
		    sg.popup_ok_cancel('Do you want to introduce another video?')
		    break



if __name__ == '__main__':
	Window_1st()
	CustomMeter()
	MachineLearningGUI()
	Output_Window()
	CustomMeter2()
	Result_Window()