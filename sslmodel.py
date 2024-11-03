""" Helper classes and functions for the SSL model """

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from transforms3d.axangles import axangle2mat
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import ipdb
import wandb
import segmentation_model

verbose = False
wandb_flag = False
torch_cache_path = Path(__file__).parent / 'torch_hub_cache'
task = 'segmentation'  # 'classification' or 'segmentation'


class RandomSwitchAxis:
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)

        return sample


class RotationAxis:
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample


class NormalDataset(Dataset):
    def __init__(self,
                 X,
                 y=None,
                 pid=None,
                 name="",
                 augmentation=False,
                 transpose_channels_first=True):

        X = X.astype(
            "f4"
        )  # PyTorch defaults to float32

        if transpose_channels_first:
            X = np.transpose(X, (0, 2, 1))
        self.X = torch.from_numpy(X)

        if y is not None:
            self.y = torch.tensor(y)
        else:
            self.y = None

        self.pid = pid

        if augmentation:
            self.transform = transforms.Compose([RandomSwitchAxis(), RotationAxis()])
        else:
            self.transform = None

        if verbose:
            print(f"{name} set sample count: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        if self.y is not None:
            y = self.y[idx]
        else:
            y = np.NaN

        if self.pid is not None:
            pid = self.pid[idx]
        else:
            pid = np.NaN

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, y, pid


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
            self,
            patience=15,
            verbose=False,
            delta=0,
            path="/mlwell-data2/dafna/ssl_gait_detection/model_outputs/checkpoints/checkpoint.pt",#"checkpoint.pt",
            trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss
        self.best_score = score
        self.save_checkpoint(val_loss, model)  
        self.counter += 1
        return  
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f" ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_sslnet(tag='v1.0.0', pretrained=False, num_classes=2, model_type='segmentation',padding_type='no_padding'):
    """
    Load and return the Self Supervised Learning (SSL) model from pytorch hub.

    :param str tag: Tag on the ssl-wearables repo to check out
    :param bool pretrained: Initialise the model with UKB self-supervised pretrained weights.
    :return: pytorch SSL model
    :rtype: nn.Module
    :model_type:'classifier', 'segmentation', 'vanila'
    """
    repo_name = 'ssl-wearables'
    repo = f'OxWearables/{repo_name}:{tag}'

    if not torch_cache_path.exists():
        Path.mkdir(torch_cache_path, parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_cache_path))

    # find repo cache dir that matches repo name and tag
    cache_dirs = [f for f in torch_cache_path.iterdir() if f.is_dir()]
    repo_path = next((f for f in cache_dirs if repo_name in f.name and tag in f.name), None)

    if repo_path is None:
        repo_path = repo
        source = 'github'
    else:
        repo_path = str(repo_path)
        source = 'local'
        if verbose:
            print(f'Using local {repo_path}')

    # if model_type == 'vanila':
    #     repo = 'OxWearables/ssl-wearables'
    #     sslnet: nn.Module = torch.hub.load(repo, 'harnet10', class_num=5, pretrained=True)
    #     return sslnet
    
    sslnet: nn.Module = torch.hub.load(repo_path, 'harnet10', trust_repo=True, source=source, class_num=num_classes,
                                       pretrained=pretrained, verbose=verbose)
    if model_type in ['classification', 'vanila']:
        return sslnet
    if model_type=='segmentation':
        seg_model = segmentation_model.SegModel(sslnet, multi_windows=padding_type=='triple_wind')
        return seg_model


def predict(model, data_loader, device):
    """
    Iterate over the dataloader and do prediction with a pytorch model.

    :param nn.Module model: pytorch Module
    :param DataLoader data_loader: pytorch dataloader
    :param str device: pytorch map device
    :param bool output_logits: When True, output the raw outputs (logits) from the last layer (before classification).
                                When False, argmax the logits and output a classification scalar.
    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    predictions_list = []
    predictions_logits_list = []
    true_list = []
    pid_list = []
    model.eval()

    for i, (x, y, pid) in enumerate(tqdm(data_loader, mininterval=60, disable=not verbose)):
        with torch.inference_mode():
            x = x.to(device, dtype=torch.float)
            logits = model(x)
            # import segmentation_model
            # seg_model = segmentation_model.SegModel(model)
            # logits2 = seg_model(x)
            # ipdb.set_trace()
            true_list.append(y)
            predictions_logits_list.append(logits.cpu())
            try:
                pred_y = torch.argmax(logits, dim=1)
            except:
                ipdb.set_trace()
            predictions_list.append(pred_y.cpu())
            pid_list.extend(pid)
    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)
    predictions_logits_list = torch.cat(predictions_logits_list)
    
    return (
        torch.flatten(true_list).numpy(),
        predictions_logits_list.numpy(),
        torch.flatten(predictions_list).numpy(),
        np.array(pid_list),
    )


def train(model, train_loader, val_loader, device, wandb_flag, is_init_estimator=True, class_weights=None, weights_path='weights.pt',
          num_epoch=30, learning_rate=0.0001, patience=25, model_type='segmentation',gait_only=False):
    """
    Iterate over the training dataloader and train a pytorch model.
    After each epoch, validate model and early stop when validation loss function bottoms out.

    Trained model weights will be saved to disk (weights_path).

    :param nn.Module model: pytorch model
    :param DataLoader train_loader: training data loader
    :param DataLoader val_loader: validation data loader
    :param str device: pytorch map device
    :param class_weights: Array of training class weights to use with weighted cross entropy loss.
                        Leave empty to use unweighted loss.
    :param weights_path: save location for the trained weights (state_dict)
    :param num_epoch: number of training epochs
    :param learning_rate: Adam learning rate
    :param patience: early stopping patience
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, amsgrad=True
    )
    if model_type =='classification':
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            loss_fn_base = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # loss_fn_base = nn.CrossEntropyLoss()
            loss_fn_base = nn.MSELoss()
        if gait_only:
            if is_init_estimator:
                loss_fn = lambda x, y : loss_fn_base(get_gait(x,is_logits=True,is_pred=True),get_gait(y)) 
            else:
                loss_fn = lambda x, y : loss_fn_base(get_gait(x,is_logits=False,is_pred=True),get_gait(y[:,:-1],is_pred=True))
        else:
            if is_init_estimator:
                loss_fn = lambda x, y : loss_fn_base(get_gait(x,is_logits=True,is_pred=True),get_gait(y)) + \
                                        10*loss_fn_base(get_valid_chorea(y)*get_chorea(x, is_logits=True,is_pred=True),get_valid_chorea(y)*get_chorea(y))
            else:
                loss_fn = lambda x, y : loss_fn_base(get_gait(x,is_logits=False,is_pred=True),get_gait(y[:,:-1],is_pred=True)) + \
                                        10*loss_fn_base(y[:,-1:]*get_chorea(x, is_logits=False, is_pred=True),y[:,-1:]*get_chorea(y[:,:-1],is_pred=True))
    


    elif model_type=='segmentation':
        def segmentaion_loss_fn(model_out, y):
            gait_labels = y[:, :, 0]
            chorea_labels = y[:, :, 1]
            gait_valid = y[:, :, 2]
            chorea_valid = y[:, :, 3]
            gait_loss = _masked_cross_entropy(gait_labels, model_out[:,0:2,:], gait_valid)
            if gait_only:
                chorea_loss = 0
            else:
                chorea_loss=_masked_cross_entropy(chorea_labels, model_out[:,2:7,:], chorea_valid)
            return gait_loss + chorea_loss
        loss_fn = segmentaion_loss_fn
    # loss_gait = loss_fn(get_gait(logits),get_gait(true_y))
    # loss_chorea = loss_fn(get_chorea(logits),get_chorea(true_y))
    # loss = loss_gait+loss_chorea


    
    early_stopping = EarlyStopping(
        patience=patience, path=weights_path, verbose=verbose, trace_func=print
    )

    gait_cross_entropy = nn.CrossEntropyLoss()
    chorea_cross_entropy = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        model.train()
        train_losses = []
        train_gait_acces = []
        train_chorea_acces = []
        for i, (x, y, _) in enumerate(tqdm(train_loader, disable=not verbose)):
            x.requires_grad_(True)
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.float)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, true_y)

            
            loss.backward()
            optimizer.step()

            train_acc_gait, train_acc_chorea = calc_gait_and_chorea_acc(true_y, logits)

            train_losses.append(loss.cpu().detach())
            train_gait_acces.append(train_acc_gait.cpu().detach())
            train_chorea_acces.append(train_acc_chorea.cpu().detach())
        if val_loader is not None:
            val_loss, gait_acc, chorea_acc = _validate_model(model, val_loader, device, loss_fn)
        else:
            val_loss=gait_acc=chorea_acc=-1

        epoch_len = len(str(num_epoch))
        print_msg = (
                f"[{epoch:>{epoch_len}}/{num_epoch:>{epoch_len}}] | "
                + f"train_loss: {np.mean(train_losses):.3f} | "
                + f"train_gait_acc: {np.mean(train_gait_acces):.3f} | "
                + f"train_chorea_acc: {np.mean(train_chorea_acces):.3f} | "
                + f"val_loss: {val_loss:.3f} | "
                + f"val_gait_acc: {gait_acc:.2f} | "
                + f"val_chorea_acc: {chorea_acc:.2f} | "
        )
        if wandb_flag:
            wandb.log({"train_loss":  np.mean(train_losses)})
            wandb.log({"val_loss":  np.mean(val_acc)})
            wandb.log({"train_gait_acc":  np.mean(train_gait_acces)})
            wandb.log({"train_chorea_acc":  np.mean(train_chorea_acces)})
            wandb.log({"val_gait_acc":  gait_acc})
            wandb.log({"val_chorea_acc":  chorea_acc})
        early_stopping(val_loss, model)

        if verbose:
            print(print_msg)

        if early_stopping.early_stop:
            if verbose:
                print('Early stopping')
                print(f'SSLNet weights saved to {weights_path}')
            break

    return model

def _masked_cross_entropy(labels, logits, mask):
    masked_logits = logits * mask.unsqueeze(1)
    masked_labels = labels.long() * mask.long()

    # Compute the cross-entropy loss
    loss = F.cross_entropy(masked_logits, masked_labels, reduction='sum')

    # Normalize the loss by the number of valid pixels
    num_valid_pixels = mask.sum().item()
    normalized_loss = loss / (num_valid_pixels + 1e-5)
    return normalized_loss

def _validate_model(model, val_loader, device, loss_fn):
    """ Iterate over a validation data loader and return mean model loss and accuracy. """
    model.eval()
    losses = []
    gait_acces = []
    chora_acces = []
    
    for i, (x, y, _) in enumerate(val_loader):
        with torch.inference_mode():
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.float)
            logits = model(x)
            loss = loss_fn(logits, true_y)
            # loss_gait = loss_fn(get_gait(logits),get_gait(true_y))
            # loss_chorea = loss_fn(get_chorea(logits),get_chorea(true_y))
            # loss = loss_gait+loss_chorea
            val_acc_gait, val_acc_chorea = calc_gait_and_chorea_acc(true_y, logits)

            gait_acces.append(val_acc_gait.cpu().detach())
            chora_acces.append(val_acc_chorea.cpu().detach())

            losses.append(loss.cpu().detach())
    losses = np.array(losses)
    return np.mean(losses), np.mean(np.array(gait_acces)), np.mean(np.array(chora_acces))

def get_gait(y, is_logits=False, is_pred=False):
    try:
        if is_logits:
            # TODO: change to softmax
            y = torch.sigmoid(y)

        if is_pred:
            class_1 = y[:,0]
            class_2 = y[:,1]
        else:
            class_1 = torch.sum(y[:, 0::2], dim=1)
            class_2 = torch.sum(y[:, 1::2], dim=1)

        return torch.stack([class_1, class_2], dim=1)
    except:
        ipdb.set_trace()
    #return torch.tensor([torch.sum(y[0:5]),torch.sum(y[5:])])
def get_chorea(y, is_logits=False,is_pred=True):
    try:
        if is_logits:
            y = torch.sigmoid(y)
        # is_pred = y.shape[-1] == 8
        if y.shape[-1] <= 10:
            return torch.stack([y[:,i] for i in range(2, 7)], dim=1)
        else:
            return torch.stack([y[:,i*2] + y[:,i*2+1] for i in range(5)], dim=1)
    except:
        ipdb.set_trace()

def get_gait_grad(x, y, is_multi_label=True):
    ''' return dL/dx for -x'''
    y_gait = get_gait(y)
    x_gait = get_gait(x)
    x_gait_prob = torch.nn.functional.softmax(x_gait, dim=1)
    if is_multi_label:
        x_gait = get_gait(x,is_pred=True)
        x_gait_prob = torch.nn.functional.softmax(x_gait, dim=1)
        return y_gait - x_gait_prob
    else:
        y_gait_repeat = y_gait.repeat([1, 5])
        x_gait_repeat = x_gait_prob.repeat([1, 5])
        return  y_gait_repeat - x_gait_repeat

def get_chorea_grad(x, y, is_multi_label=True):
    ipdb.set_trace()
    y_chorea = get_chorea(y)
    valid_chorea = get_valid_chorea(y)
    x_chorea = get_chorea(x)
    x_chorea_prob = torch.nn.functional.softmax(x_chorea, dim=1)
    if is_multi_label:
        x_chorea = get_chorea(x,is_pred=True)
        x_chorea_prob = torch.nn.functional.softmax(x_chorea, dim=1)
        return valid_chorea*(y_chorea - x_chorea_prob)
    else:
        y_chorea_repeat = y_chorea.repeat_interleave(2,dim=1)
        x_chorea_repeat = x_chorea_prob.repeat_interleave(2,dim=1)
        return valid_chorea*(y_chorea_repeat - x_chorea_repeat)
    #return torch.tensor([y[i]+y[5+i] for i in range(5)])
def get_valid_chorea(y):
    return torch.unsqueeze(torch.sum(y[:, :10], dim=1)==1, axis=-1)

def calc_gait_and_chorea_acc(true_y, pred_y):
    if task == 'segmentation':
        gait_pred = torch.argmax(pred_y[:, 0:2, :], axis=1)
        gait_valid = true_y[:, :, 2]
        gait_label = true_y[:, :, 0]
        gait_match = (gait_label == gait_pred) * gait_valid
        val_acc_gait = torch.sum(gait_match) / (torch.sum(gait_valid) + 1e-5)

        chorea_pred = torch.argmax(pred_y[:, 2:7, :], axis=1)
        chorea_valid = true_y[:, :, 3]
        chorea_label = true_y[:, :, 1]
        chorea_match = (chorea_label == chorea_pred) * chorea_valid
        val_acc_chorea = torch.sum(chorea_match) / (torch.sum(chorea_valid) + 1e-5)
        return val_acc_gait, val_acc_chorea
    else:
        pred_y_gait = get_gait(pred_y,is_pred=True)
        pred_y_chorea = get_chorea(pred_y,is_pred=True)
        true_y_gait = get_gait(true_y, is_pred=False)
        true_y_chorea = get_chorea(true_y, is_pred=False)
        valid_chorea = get_valid_chorea(true_y)

        pred_gait = torch.argmax(pred_y_gait, dim=1)
        pred_chorea = torch.argmax(valid_chorea*pred_y_chorea, dim=1)
        true_gait = torch.argmax(true_y_gait, dim=1)
        true_chorea = torch.argmax(valid_chorea*true_y_chorea, dim=1)
        val_acc_gait = torch.sum(pred_gait == true_gait)
        val_acc_chorea = torch.sum(pred_chorea == true_chorea) - torch.sum(torch.logical_not(valid_chorea))
        val_acc_gait = val_acc_gait/(list(pred_y.size())[0])
        val_acc_chorea = val_acc_chorea/(torch.sum(valid_chorea)+1e-7)
        return val_acc_gait, val_acc_chorea
