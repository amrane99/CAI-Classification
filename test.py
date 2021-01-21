from train_restore_use_models.Classification_train_restore_use import Classification_initialize_and_train


config = {'device':'cuda:6', 'nr_runs': 1, 'cross_validation': False, 
          'val_ratio': 0.2, 'test_ratio': 0.2, 'input_shape': (3, 480, 854),
          'resize': False, 'augmentation': 'none', 'lr': 0.001, 'batch_size': 64,
          'number_of_tools': 8, 'nr_epochs': 2, 
          'random_slices': True, 'nr_videos': 40, 'nr_slices': 25,
          'weight_decay': 0.75, 'save_interval': 25, 'msg_bot': False,
          'bot_msg_interval': 20, 'augmented': True, 'dataset': 'Cholec80'
         }

Classification_initialize_and_train(config)