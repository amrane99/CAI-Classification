# ------------------------------------------------------------------------------
# A classification agent.
# ------------------------------------------------------------------------------

from cai.agents.agent import Agent
from cai.paths import telegram_login
from cai.utils.update_bots.telegram_bot import TelegramBot

class ClassificationAgent(Agent):
    r"""An Agent for classification models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot(telegram_login)

    def train(self, optimizer, loss_f, train_dataloader,
              val_dataloader, nr_epochs=100, save_path=None,
              save_interval=10, msg_bot=True, bot_msg_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        # TODO
        pass

    def test(self, loss_f, test_dataloader, msg_bot=True):
        # TODO
        pass
