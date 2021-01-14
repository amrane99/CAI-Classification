# ------------------------------------------------------------------------------
# An Agent is an extension of a model. It also includes the logic to train the 
# model. This superclass diverges slightly from cai.agents.agent.Agent.
# Substantial portions from https://github.com/camgbus/medical_pytorch project.
# ------------------------------------------------------------------------------

import os
import shutil
from cai.utils.load_restore import pkl_dump, pkl_load
from cai.utils.pytorch.pytorch_load_restore import save_model_state, load_model_state, save_optimizer_state, load_optimizer_state

class Agent:
    r"""An Agent, which includes a model and extended fields and logic.

    Args:
        model (mp.models.model.Model): a model
        label_names (list[str]): a list of label names
        device (str): 'cpu' or a cuda-enabled gpu, e.g. 'cuda'
    """
    def __init__(self, model, label_names=None, device='cuda'):
        self.model = model
        self.device = device
        self.label_names = label_names
        self.nr_labels = len(label_names) if self.label_names else 0
        self.agent_state_dict = dict()

    def train(self, optimizer, loss_f, train_dataloader,
              val_dataloader, nr_epochs=100, save_path=None,
              save_interval=10, msg_bot=False, bot_msg_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states. Needs to be overwritten.
        """
        pass

    def test(self, loss_f, test_dataloader, msg_bot=False):
        r"""Test a model through its agent. Needs to be overwritten.
        """
        pass

    def save_state(self, states_path, state_name, optimizer=None, overwrite=False):
        r"""Saves an agent state. Raises an error if the directory exists and 
        overwrite=False.
        """
        if states_path is not None:
            state_full_path = os.path.join(states_path, state_name)
            if os.path.exists(state_full_path):
                if not overwrite:
                    raise FileExistsError
                shutil.rmtree(state_full_path)
            os.makedirs(state_full_path)
            save_model_state(self.model, 'model', state_full_path)
            pkl_dump(self.agent_state_dict, 'agent_state_dict', state_full_path)
            if optimizer is not None:
                save_optimizer_state(optimizer, 'optimizer', state_full_path)

    def restore_state(self, states_path, state_name, optimizer=None):
        r"""Tries to restore a previous agent state, consisting of a model 
        state and the content of agent_state_dict. Returns whether the restore 
        operation  was successful.
        """
        state_full_path = os.path.join(states_path, state_name)
        try:
            correct_load = load_model_state(self.model, 'model', state_full_path, device=self.device)
            assert correct_load
            agent_state_dict = pkl_load('agent_state_dict', state_full_path)
            assert agent_state_dict is not None
            self.agent_state_dict = agent_state_dict
            if optimizer is not None: 
                load_optimizer_state(optimizer, 'optimizer', state_full_path, device=self.device)
            if self.verbose:
                print('State {} was restored'.format(state_name))
            return True
        except:
            print('State {} could not be restored'.format(state_name))
            return False