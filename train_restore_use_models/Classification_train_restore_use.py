# Import needed libraries
import torch
import os
import numpy as np
import pandas as pd
import shutil
from cai.paths import model_result_path, storage_data_path
from torch.utils.data import DataLoader
import torch.optim as optim
from cai.data.data import Data
from cai.data.datasets.ds_cholec80_classification import Cholec80, Cholec80Restored
from cai.data.datasets.data_splitting import split_dataset
import cai.utils.load_restore as lr
from cai.data.pytorch.pytorch_classification_dataset import PytorchClassification2DDataset
import cai.models.classification.CNN as models
from cai.eval.losses.losses_classification import LossBCE
import cai.agents.classification_agents as agents
from cai.utils.save_results import save_results, save_only_test_results
import cai.gui.gui as gui
import moviepy.editor as mpy
import time
import sys
from cai.utils.load_restore import join_path
import PySimpleGUI as sg
import json
from torchvision import transforms

# Labels:
# 0 = Grasper
# 1 = Bipolar
# 2 = Hook
# 3 = Scissors
# 4 = Clipper
# 5 = Irrigator
# 6 = Specimenbag

def Classification_initialize_and_train(config):
    r"""This function selects random images etc. based on the config file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    number_of_tools = config['number_of_tools']
    output_features = number_of_tools
    random_frames = config['random_frames']
    nr_videos = config['nr_videos']
    nr_frames = config['nr_frames']
    weight_decay = config['weight_decay']
    save_interval = config['save_interval']
    msg_bot = config['msg_bot']
    bot_msg_interval = config['bot_msg_interval']
    dataset_name = config['dataset']
    model_name = config['model']
    agent_name = 'TransNetAgent'


    # 2. Define data
    data = Data()
    data.add_dataset(Cholec80(random_frames, nr_videos, nr_frames))
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')


    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
        val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])

    # Include the model name, Alexnet, CNN, Resnet etc. what has been used
    paths = os.path.join(storage_data_path, 'models', dataset_name+'_'+model_name, 'states')
    pathr = os.path.join(model_result_path, 'models', dataset_name+'_'+model_name, 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    else:
        # Empty directory
        shutil.rmtree(paths)
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # Save split
    if splits is not None:
        lr.save_json(splits, path=paths, name='data_splits')


    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 5. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchClassification2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])   #TODO: Test with resize=True and without approach from dataset_classification.py

        # 6. Build train dataloader, and visualize
        dl = DataLoader(datasets[(train_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)
        dl_val = DataLoader(datasets[(val_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Initialize model
        model = getattr(models, model_name)(output_features)
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossBCE(device=device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=weight_decay)

        # 9. Train model
        print('Training model in batches of {}..'.format(batch_size))

        agent = getattr(agents, agent_name)(model=model, device=device)
        losses_train, losses_cum_train, losses_val, losses_cum_val,\
        accuracy_train, accuracy_det_train, accuracy_val,\
        accuracy_det_val = agent.train(optimizer, loss_f, dl,
                       dl_val, nr_epochs=config['nr_epochs'],
                                             save_path=paths,
                save_interval=save_interval, msg_bot=msg_bot,
                           bot_msg_interval=bot_msg_interval)
                           
        # 10. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 11. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 12. Save results
    save_results(model, model_name, dataset_name, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)
    
def Classification_restore_and_train(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    number_of_tools = config['number_of_tools']
    output_features = number_of_tools
    random_frames = config['random_frames']
    nr_videos = config['nr_videos']
    nr_frames = config['nr_frames']
    weight_decay = config['weight_decay']
    save_interval = config['save_interval']
    msg_bot = config['msg_bot']
    bot_msg_interval = config['bot_msg_interval']
    dataset_name = config['dataset']
    model_name = config['model']
    agent_name = 'TransNetAgent'


    # 2. Define data to restore dataset
    data = Data()
    data.add_dataset(Cholec80Restored())
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')


    # 3. Restore and define path
    paths = os.path.join(storage_data_path, 'models', dataset_name+'_'+model_name, 'states')
    pathr = os.path.join(model_result_path, 'models', dataset_name+'_'+model_name, 'results')
    splits = lr.load_json(path=paths, name='data_splits')
    print('Restored existing splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 5. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchClassification2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])

        # 6. Build train dataloader, and visualize
        dl = DataLoader(datasets[(train_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=0)
        dl_val = DataLoader(datasets[(val_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Initialize model
        model = getattr(models, model_name)(output_features)
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossBCE(device=device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=weight_decay)

        # 9. Train model
        state_names = [name for name in os.listdir(paths) if '.' not in name]
        state_name = state_names[0].split('_')[0]
        for idx, state in enumerate(state_names):
            state_names[idx] = int(state.split('_')[-1])
        state_names.sort()
        state_name += '_' + str(state_names[-1])

        print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
        agent = getattr(agents, agent_name)(model=model, device=device)
        restored, restored_results = agent.restore_state(paths, state_name, optimizer=optimizer)
        if not restored:
            print("Desired state could not be recovered. --> Error!")
            raise FileNotFoundError

        losses_train_r, losses_cum_train_r, losses_val_r, losses_cum_val_r, accuracy_train_r,\
        accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r = restored_results

        print('Training model in batches of {}..'.format(batch_size))
        losses_train, losses_cum_train, losses_val, losses_cum_val,\
        accuracy_train, accuracy_det_train, accuracy_val,\
        accuracy_det_val = agent.train(optimizer, loss_f, dl,
                       dl_val, nr_epochs=config['nr_epochs'],
                  start_epoch=int(state_name.split('_')[-1]),
             save_path=paths, losses=losses_train_r.tolist(),
                      losses_cum=losses_cum_train_r.tolist(),
                            losses_val=losses_val_r.tolist(),
                    losses_cum_val=losses_cum_val_r.tolist(),
                          accuracy=accuracy_train_r.tolist(),
             accuracy_detailed=accuracy_det_train_r.tolist(),
                        accuracy_val=accuracy_val_r.tolist(),
           accuracy_val_detailed=accuracy_det_val_r.tolist(),
                save_interval=save_interval, msg_bot=msg_bot,
                           bot_msg_interval=bot_msg_interval)

        # 10. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 11. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 12. Save results
    save_results(model, model_name, dataset_name, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)

def Classification_test(config):
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file
        and evaluates the output."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size']
    number_of_tools = config['number_of_tools']
    output_features = number_of_tools
    random_frames = config['random_frames']
    nr_videos = config['nr_videos']
    nr_frames = config['nr_frames']
    weight_decay = config['weight_decay']
    msg_bot = config['msg_bot']
    dataset_name = config['dataset']
    model_name = config['model']
    agent_name = 'TransNetAgent'


    # 2. Define data
    data = Data()
    data.add_dataset(Cholec80(random_frames, nr_videos, nr_frames))
    test_ds = (dataset_name, 'test')


    # 3. Split data (0% train, 100% test) and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=1.0, 
        val_ratio=0, nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    pathr = os.path.join(model_result_path, 'models', dataset_name+'_'+model, 'test_results')
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)


    # 4. Bring data to Pytorch format
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 5. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchClassification2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])
                
        # 6. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Load pretrained model
        model = torch.load(os.path.join(model_result_path, 'models_', model_name, 'model.zip'))
        model.eval()
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossBCE(device=device)
        
        # 9. Test model
        agent = getattr(agents, agent_name)(model=model, device=device)
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, _, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 10. Save results
    save_only_test_results(pathr, losses_test, accuracy_test, accuracy_det_test)

def Classification_predict():
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file
    in an interactive way using the implemented GUI."""
    
    # Config dicts for each possible model and tool list for mapping
    config_AlexNet = {'nr_epochs': 300, 'batch_size': 32, 'learning_rate': 0.0001,
                      'weight_decay': 0.001, 'model': 'AlexNet', 'test_acc': 80}
    config_ResNet =  {'nr_epochs': 300, 'batch_size': 32, 'learning_rate': 0.0001,
                      'weight_decay': 0.001, 'model': 'ResNet', 'test_acc': 80}
    tool = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Specimenbag']
    start_again = False
    prev_video_path = ''
    prev_target_path = ''
    prev_model_names = 'AlexNet'
    prev_cpu = True
    prev_gpu = False
    prev_start_id = 0
    
    # !!Still need to be implemented with the backwards step!!
    while True:
        # 2. Display Welcome and Start Frame
        welcome = gui.WelcomeWindow()
        if not welcome:
            sys.exit()
        
        while True:
            start = gui.StartWindow(prev_video_path)
            if not start[0]:
                sys.exit()
                
            # 3. Load video if path has been changed after backward step
            video_path = start[1]
            filename = video_path.split('/')[-1].split('.')[0]
            load_window, load_progress_bar = gui.LoadVideo()
            
            _, _ = load_window.read(timeout=0, timeout_key="timeout")
            if prev_video_path == video_path:
                print('Video already in memory..')
                load_progress_bar.update_bar(1)
            else:
                print('Loading the video..')
                video = mpy.VideoFileClip(video_path)
                load_progress_bar.update_bar(1)
            
            # 4. Extract video features
            fps = int(video.fps)
            video_length = video.duration
            frames = int(video.fps * video.duration)-2
            size = str((video.w, video.h, 3))
            
            # Transform video length from seconds into string with format h - m - s
            result = time.strftime('%H:%M:%S', time.gmtime(video_length)).split(':')
            length = result[0] + ' h - ' + result[1] + ' m - ' + str(int(result[2])-2) + ' s'
            load_progress_bar.update_bar(2)
            load_window.CloseNonBlocking()
            
            # Build video information dict
            video_info = {"path": str(video_path), "frames": str(frames),
                          "fps": str(fps), "length": str(length),
                          "size": str(size)}
             
            while True:
                # 5. Build video information file
                transform = gui.TransformVideo(prev_target_path, video_info)
                if not transform[0]:
                    sys.exit()
                if transform[2]:
                    break
                
                # Transform the video and save it if needed
                target_path = transform[1]
                
                if prev_target_path == target_path and os.path.isfile(os.path.join(target_path, filename+'_transformed.mp4')) and int(video.fps) == 1 and int(video.h) == 224 and int(video.w) == 224:
                    print('Video already transformed..')
                elif target_path is not None and fps != 1:
                    print('Transforming the video..')
                    transform_window, transform_progress_bar = gui.TransformVideoProgress()
                    _, _ = transform_window.read(timeout=0, timeout_key="timeout")
                    # Reduce fps to 1 and extract video properties
                    video = video.set_fps(1)
                    transform_progress_bar.update_bar(1)
                    # Resize frames to 224x224
                    video = video.resize((224,224))
                    transform_progress_bar.update_bar(2)
                    # Save transformed video --> Works, since target path does already exist at this point
                    video.write_videofile(join_path([target_path, filename+'_transformed.mp4']))
                    # Update number of frames
                    frames = int(video.fps * video.duration)-2
                    transform_progress_bar.update_bar(3)
                    transform_window.CloseNonBlocking()
                    
                while True:
                    # 6. Let user choose model and device
                    model_names = ('AlexNet', 'ResNet')
                    model_device = gui.ChooseModelAndDevice(prev_model_names, model_names, prev_cpu, prev_gpu, prev_start_id)
                    if not model_device[0]:
                        sys.exit()
                    if model_device[2]:
                        break
                        
                    # Extract model name and GPU device ID/CPU
                    model_name = model_device[1][0]
                    
                    if model_device[1][1] is not None:
                        device = 'cuda:' + str(model_device[1][1])
                        prev_cpu = False
                        prev_gpu = True
                        prev_start_id = model_device[1][1]
                    else:
                        device = 'cpu'
                        prev_cpu = True
                        prev_gpu = False
                        
                    while True:
                        # 7. Display model specifications based on model name
                        if model_name == 'AlexNet':
                            model_specs = gui.ModelSpecs(config_AlexNet)
                        if not model_specs[0]:
                            sys.exit()
                        if model_specs[1]:
                            break
                            
                        # 8. Make predictions based on users inputs
                        prediction_window, prediction_progress_bar = gui.PredictVideoTools(frames+2)
                        _, _ = prediction_window.read(timeout=0, timeout_key="timeout")
                            
                        # Load pre-trained model structure
                        model_result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'models')
                        if device == 'cpu':
                            model = torch.load(os.path.join(model_result_path, 'Cholec80_' + model_name, 'model.zip'), map_location=torch.device('cpu'))
                        else:
                            model = torch.load(os.path.join(model_result_path, 'Cholec80_' + model_name, 'model.zip'))
                        model.eval()
                        model.to(device)
                        prediction_progress_bar.update_bar(1)
                        
                        # Make predictions
                        print('Predicting tools in the video..')
                            
                        predictions = dict()
                        normalize = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        
                        for idx in range(frames):
                            msg = "Predicting present tools in Frame "
                            msg += str(idx + 1) + " of " + str(frames) + "."
                            print (msg, end = "\r")
                            np_frame = video.get_frame(idx)
                            x = torch.from_numpy(np_frame).permute(2, 0, 1)
                            # Normalization only necessary for transfer learned models
                            x = normalize(x.cpu().detach())
                            yhat = model(x.unsqueeze(0))
                            yhat = yhat.cpu().detach().numpy()
                            frame_to_sec = time.strftime('%H:%M:%S', time.gmtime(idx+1)).split(':')
                            frame_sec = frame_to_sec[0] + ' h - ' + frame_to_sec[1] + ' m - ' + str(int(frame_to_sec[2])) + ' s'
                            predictions['Frame ' + str(idx+1) + ' --> ' + str(frame_sec) + ':'] = np.round(yhat)[0]
                            prediction_progress_bar.update_bar(idx+2)
                        
                        # 9. Transform predictions and save the result under target path
                        result = ''
                        for key, value in predictions.items():
                            # Transform vector
                            tools = ''
                            for idx, i in enumerate(value):
                                if i == 1:
                                    tools += tool[idx] + ', '
                            # Replace value from dict with transformed one
                            predictions[key] = tools[:-2]
                            result += str(key) + ' ' + str(tools[:-2] + '\n')
                            
                        prediction_progress_bar.update_bar(frames+2)
                        prediction_window.CloseNonBlocking()
                        
                        # 10. Print the results
                        results = gui.ResultWindow(result)
                        
                        # 11. Save result as json in target_paths
                        print("Saving results in form of .json..")
                        with open(os.path.join(target_path, filename+'-tool.json'), 'w') as fp:
                            json.dump(predictions, fp, sort_keys=False, indent=4)
                            
                        if not results:
                            sys.exit()
                        start_again = True
                        break
                            
                    prev_model_names = model_name
                    if start_again:
                        break
                prev_target_path = target_path
                if start_again:
                    break
            prev_video_path = video_path
