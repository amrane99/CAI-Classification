# ------------------------------------------------------------------------------
# A group of cnn/classification agents.
# ------------------------------------------------------------------------------

from cai.agents.agent import Agent
import torch
import os
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from cai.paths import telegram_login
from cai.utils.update_bots.telegram_bot import TelegramBot

class TransNetAgent(Agent):
    r"""An Agent for transfer learning models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot(telegram_login)

    def preprocess(self, img_tensor):
        r"""Transforms an image based on the desired input of
        transfer model.
        """
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        batch_size = list(img_tensor.size())[0]
        new_img = preprocess(img_tensor[0].cpu().detach()).unsqueeze(0)
        for batch in range(1, batch_size):
            img = img_tensor[batch]
            img = preprocess(img.cpu().detach()).unsqueeze(0) # preprocess image and add batch dimension
            new_img = torch.cat((new_img, img), 0)
        return new_img.to(self.device)

    def train(self, optimizer, loss_f, train_dataloader,
              val_dataloader, nr_epochs=100, start_epoch=0, save_path=None,
              losses=list(), losses_cum=list(), losses_val=list(), losses_cum_val=list(),
              accuracy=list(), accuracy_detailed=list(), accuracy_val=list(),
              accuracy_val_detailed=list(), save_interval=10,
              msg_bot=True, bot_msg_interval=10):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.
        """
        assert start_epoch < nr_epochs, 'Start epoch needs to be smaller than the number of epochs!'
        if msg_bot == True:
            self.bot.send_msg('Start training the ResNet model for {} epochs..'.format(nr_epochs-start_epoch))

        for epoch in range(start_epoch, nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            results_y = list()
            results_yhat = list()
            total = 0
            acc = 0
            idxs = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                x = self.preprocess(x)
                yhat = self.model(x)
                loss = loss_f(yhat, y.float())
                total += y.size(0)
                mod_yhat = np.round(yhat.cpu().detach())
                acc += 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat)
                idxs += 1
                epoch_loss.append(loss.item())
                results_y.extend(y.cpu().detach().numpy().tolist())
                results_yhat.extend(mod_yhat.cpu().detach().numpy().tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            accuracy.append([epoch+1, acc / idxs])
            accuracy_detailed.append(list(zip(results_y, results_yhat)))

            # Validate current model based on validation dataloader
            epoch_loss_val = list()
            results_y_val = list()
            results_yhat_val = list()
            results_mod_yhat_val = list()
            total_val = 0
            acc_val = 0
            idxs_val = 0
            with torch.no_grad():
                for idx, (x, y) in enumerate(val_dataloader):
                    x_val, y_val = x.to(self.device), y.to(self.device)
                    x_val = self.preprocess(x_val)
                    yhat_val = self.model(x_val)
                    loss = loss_f(yhat_val, y_val.float())
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                    mod_yhat_val = np.round(yhat_val.cpu().detach())
                    acc_val += 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat_val)
                    idxs_val += 1
                    results_y_val.extend(y_val.cpu().detach().numpy().tolist())
                    results_yhat_val.extend(mod_yhat_val.cpu().detach().numpy().tolist())
                losses_val.append(epoch_loss_val)
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])
                accuracy_val.append([epoch+1, acc_val / idxs_val])
                accuracy_val_detailed.append(list(zip(results_y_val, results_yhat_val)))

            print(('ResNet: Epoch --> Loss --> Accuracy: {} --> {:.4} --> {:.4}%.\n'
                   'ResNet: Val_Loss --> Val_Accuracy: {:.4} --> {:.4}%.').format(epoch + 1,
                                                    sum(epoch_loss) / total,
                                                    acc / idxs,
                                                    sum(epoch_loss_val) / total_val,
                                                    acc_val / idxs_val))
            if (epoch + 1) % bot_msg_interval == 0 and msg_bot:
                self.bot.send_msg(('Epoch --> Loss --> Accuracy: {} --> {:.4} --> {:.4}%.\n'
                                   'Val_Loss --> Val_Accuracy: {:.4} --> {:.4}%.').format(epoch + 1,
                                                                    sum(epoch_loss) / total,
                                                                    acc / idxs,
                                                                    sum(epoch_loss_val) / total_val,
                                                                    acc_val / idxs_val))
            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                print('Saving current state after epoch: {}.'.format(epoch + 1))
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                optimizer, True, losses, losses_cum, losses_val,
                                losses_cum_val, accuracy, accuracy_detailed, accuracy_val,
                                accuracy_val_detailed)

        # Return losses
        return losses, losses_cum, losses_val, losses_cum_val, accuracy, accuracy_detailed, accuracy_val, accuracy_val_detailed

    def test(self, loss_f, test_dataloader, msg_bot=True):
        if msg_bot == True:
            self.bot.send_msg('Start testing the ResNet model..')
        losses = list()
        accuracy = list()
        accuracy_detailed = list()
        total = 0
        losses_cum = 0
        acc = 0
        idxs = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                x = self.preprocess(x)
                yhat = self.model(x)
                loss = loss_f(yhat, y.float())
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
                mod_yhat = np.round(yhat.cpu().detach())
                acc += 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat)
                idxs += 1
                accuracy.append([idx+1, 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat)])
                accuracy_detailed.extend(list(zip(y.cpu().detach().numpy().tolist(),
                                                  mod_yhat.cpu().detach().numpy().tolist())))
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the model on the test set: %d %%' % (acc / idxs))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            self.bot.send_msg('Accuracy of the ResNet model on the test set: %d %%' % (acc / idxs))

        # Return losses
        return losses, losses_cum, accuracy, accuracy_detailed

class ClassificationAgent(Agent):
    r"""An Agent for CNN models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot(telegram_login)

    def train(self, optimizer, loss_f, train_dataloader,
              val_dataloader, nr_epochs=100, start_epoch=0, save_path=None,
              losses=list(), losses_cum=list(), losses_val=list(), losses_cum_val=list(),
              accuracy=list(), accuracy_detailed=list(), accuracy_val=list(),
              accuracy_val_detailed=list(), save_interval=10,
              msg_bot=True, bot_msg_interval=10):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.
        """
        assert start_epoch < nr_epochs, 'Start epoch needs to be smaller than the number of epochs!'
        if msg_bot == True:
            self.bot.send_msg('Start training the model for {} epochs..'.format(nr_epochs-start_epoch))

        for epoch in range(start_epoch, nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            results_y = list()
            results_yhat = list()
            total = 0
            acc = 0
            idxs = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y.float())
                total += y.size(0)
                mod_yhat = np.round(yhat.cpu().detach())
                acc += 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat)
                idxs += 1
                epoch_loss.append(loss.item())
                results_y.extend(y.cpu().detach().numpy().tolist())
                results_yhat.extend(mod_yhat.cpu().detach().numpy().tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            accuracy.append([epoch+1, acc / idxs])
            accuracy_detailed.append(list(zip(results_y, results_yhat)))

            # Validate current model based on validation dataloader
            epoch_loss_val = list()
            results_y_val = list()
            results_yhat_val = list()
            results_mod_yhat_val = list()
            total_val = 0
            acc_val = 0
            idxs_val = 0
            with torch.no_grad():
                for idx, (x, y) in enumerate(val_dataloader):
                    x_val, y_val = x.to(self.device), y.to(self.device)
                    yhat_val = self.model(x_val)
                    loss = loss_f(yhat_val, y_val.float())
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                    mod_yhat_val = np.round(yhat_val.cpu().detach())
                    acc_val += 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat_val)
                    idxs_val += 1
                    results_y_val.extend(y_val.cpu().detach().numpy().tolist())
                    results_yhat_val.extend(mod_yhat_val.cpu().detach().numpy().tolist())
                losses_val.append(epoch_loss_val)
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])
                accuracy_val.append([epoch+1, acc_val / idxs_val])
                accuracy_val_detailed.append(list(zip(results_y_val, results_yhat_val)))

            print(('Epoch --> Loss --> Accuracy: {} --> {:.4} --> {:.4}%.\n'
                   'Val_Loss --> Val_Accuracy: {:.4} --> {:.4}%.').format(epoch + 1,
                                                    sum(epoch_loss) / total,
                                                    acc / idxs,
                                                    sum(epoch_loss_val) / total_val,
                                                    acc_val / idxs_val))
            if (epoch + 1) % bot_msg_interval == 0 and msg_bot:
                self.bot.send_msg(('Epoch --> Loss --> Accuracy: {} --> {:.4} --> {:.4}%.\n'
                                   'Val_Loss --> Val_Accuracy: {:.4} --> {:.4}%.').format(epoch + 1,
                                                                    sum(epoch_loss) / total,
                                                                    acc / idxs,
                                                                    sum(epoch_loss_val) / total_val,
                                                                    acc_val / idxs_val))
            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                print('Saving current state after epoch: {}.'.format(epoch + 1))
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                optimizer, True, losses, losses_cum, losses_val,
                                losses_cum_val, accuracy, accuracy_detailed, accuracy_val,
                                accuracy_val_detailed)

        # Return losses
        return losses, losses_cum, losses_val, losses_cum_val, accuracy, accuracy_detailed, accuracy_val, accuracy_val_detailed

    def test(self, loss_f, test_dataloader, msg_bot=True):
        if msg_bot == True:
            self.bot.send_msg('Start testing the model..')
        losses = list()
        accuracy = list()
        accuracy_detailed = list()
        total = 0
        losses_cum = 0
        acc = 0
        idxs = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y.float())
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
                mod_yhat = np.round(yhat.cpu().detach())
                acc += 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat)
                idxs += 1
                accuracy.append([idx+1, 100 * accuracy_score(y.cpu().detach().numpy(), mod_yhat)])
                accuracy_detailed.extend(list(zip(y.cpu().detach().numpy().tolist(),
                                                  mod_yhat.cpu().detach().numpy().tolist())))
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the model on the test set: %d %%' % (acc / idxs))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            self.bot.send_msg('Accuracy of the model on the test set: %d %%' % (acc / idxs))

        # Return losses
        return losses, losses_cum, accuracy, accuracy_detailed
