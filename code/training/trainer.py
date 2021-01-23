import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from stud.utilities import save_to_pickle

class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        label_vocab,
        options,
        log_steps:int=128,
        log_level:int=2,
        device:str="cuda"):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
            opts: dictionary that specifies various training options and hyperparameters
            log_steps: print progess after how many steps
            log_level: print every step or every log_steps
            device: device to train on
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab

        self.device = options["device"]

        self.bert = options["use_bert"]
        self.use_pos = options["use_pos_embeddings"]

        self.clip = options["grad_clipping"]
        self.early_stop = options["early_stopping"]

        self.opts = options

    def train(self, train_dataset, 
              valid_dataset):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            valid_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.

        Returns:
            train_loss_plot: the average training loss on train_dataset over
                epochs.
        """
        if self.log_level > 0:
            print('Training ...')
        
        train_loss = 0.0

        epochs = self.opts["epochs"]
        save_folder = self.opts["save_model_path"]
        
        # Wether or not Early Stop event happened
        early_stop_event = False
        es_epoch = None

        prev_loss = list()

        pos = None
        mask = None

        train_loss_plot = list()

        for epoch in range(epochs):
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            epoch_acc = 0.0
            self.model.train()

            for step, batch in enumerate(train_dataset):

                inputs = batch["inputs"].to(self.device)
                if self.opts["use_binary_pred"] == True:
                    pred = batch["preds"].type(torch.FloatTensor).to(self.device)
                else:
                    pred = batch["preds"].to(self.device)
                labels = batch["labels"].to(self.device, dtype=torch.int64)

                if self.bert == True:
                    mask = batch["mask"].to(self.device, dtype=torch.uint8)                
                if self.use_pos == True:
                    pos = batch["pos"].to(self.device)


                self.optimizer.zero_grad()
                
                # Forward pass through the network
                predictions = self.model(inputs, mask, pred, pos)
                print("shape out of the netowrk", predictions.shape)
                predictions = predictions.view(-1, predictions.shape[-1])
                print("predictions shape after reshape", predictions.shape)
                print("labels shape BEFORE reshape", labels.shape)
                labels = labels.view(-1)
                print("labels shape after reshape", labels.shape)
                batch_loss = self.loss_function(predictions, labels)


                batch_loss.backward()

                if self.clip == True:
                    # Apply Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) 
                
                self.optimizer.step()

                epoch_loss += batch_loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))
            
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            train_loss_plot.append(avg_epoch_loss)

            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss = self.evaluate(valid_dataset)
            prev_loss.append(valid_loss)

            if self.log_level > 0:
                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

            # If we want to use early stopping
            if self.early_stop == True:
                if epoch > 0:
                    if  valid_loss > prev_loss[epoch-1]:
                        print("Validation loss increased ! Stopping training...")
                        early_stop_event = True
                        es_epoch = epoch
                        break

            # Save the model If early stopping is off 
            # or the val loss did NOT increase / Early Stop event did NOT happen
            if early_stop_event == False:
                print("Saving model")
                torch.save(self.model.state_dict(), os.path.join(save_folder, 'state_{}.pth'.format(epoch))) # save the model state
                torch.save(self.model, os.path.join(save_folder, 'checkpoint_{}.pt'.format(epoch))) # save the model state
            else:
                print("Early stop event triggered, not saving the model for this epoch")

        epochs_plot = [e for e in range(self.opts["epochs"])]
        plt.plot(train_loss_plot, "o-", color="blue", label="Training Loss")
        plt.plot(prev_loss, "o-",color="red", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig("{}/{}".format(self.opts["save_model_path"], "Loss_Plot"))
        # Save loss evolution for debugging purposes
        save_to_pickle("{}/{}".format(self.opts["save_model_path"], "Train_Loss"), train_loss_plot)
        save_to_pickle("{}/{}".format(self.opts["save_model_path"], "Val_Loss"), prev_loss)

        if self.log_level > 0:
            print('... Done!')
        
        return train_loss_plot
    

    def evaluate(self, valid_dataset):
        """
        Args:
            valid_dataset: the dataset to use to evaluate the model.

        Returns:
            avg_valid_loss: the average validation loss over valid_dataset.
        """
        valid_loss = 0.0
        
        pos = None
        pred = None
        mask = None

        # set dropout to . Needed when we are in inference mode.
        self.model.eval()
        with torch.no_grad():
            for batch in valid_dataset:

                inputs = batch["inputs"].to(self.device)
                if self.opts["use_binary_pred"] == True:
                    pred = batch["preds"].type(torch.FloatTensor).to(self.device)
                else:
                    pred = batch["preds"].to(self.device)
                labels = batch["labels"].to(self.device, dtype=torch.int64)
                if self.bert == True:
                    mask = batch["mask"].to(self.device, dtype=torch.uint8)                
                if self.use_pos == True:
                    pos = batch["pos"].to(self.device)

                predictions = self.model(inputs, mask, pred, pos)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                batch_loss = self.loss_function(predictions, labels)
                

                valid_loss += batch_loss.tolist()

                labels_list = labels.tolist()
                predictions_list = (torch.argmax(predictions, -1)).tolist()
        
        return valid_loss / len(valid_dataset)


