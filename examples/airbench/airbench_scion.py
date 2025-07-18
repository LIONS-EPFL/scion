"""
airbench94_spectral.py
Runs in 2.67 seconds on a 400W NVIDIA A100
Attains 94.04 mean accuracy (n=200 trials)
"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import math
from math import ceil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import wandb 
from scion import Scion

# ADDED
import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.backends.cudnn.benchmark = True

# We express the main training hyperparameters (batch size, learning rate, momentum, and weight decay)
# in decoupled form, so that each one can be tuned independently. This accomplishes the following:
# * Assuming time-constant gradients, the average step size is decoupled from everything but the lr.
# * The size of the weight decay update is decoupled from everything but the wd.
# In constrast, normally when we increase the (Nesterov) momentum, this also scales up the step size
# proportionally to 1 + 1 / (1 - momentum), meaning we cannot change momentum without having to re-tune
# the learning rate. Similarly, normally when we increase the learning rate this also increases the size
# of the weight decay, requiring a proportional decrease in the wd to maintain the same decay strength.
#
# The practical impact is that hyperparameter tuning is faster, since this parametrization allows each
# one to be tuned independently. See https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/.

hyp = {
    'meta': {
        'runs': 5,
    },
    'opt': {
        'svd_backend': 'newton',
        'train_epochs': 8,
        'batch_size': 2000,
        'lr': 6.5,                 # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.015,     # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 0,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net(widths):
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net


def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()
        # if type(m) in (Conv,):
        #     conv_init(m.weight)
        # if type(m) in (nn.Linear,):
        #     embedding_init(m.weight)
            #linear_init(m.weight)


#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(run, model_trainbias, model_freezebias, lr, momentum):
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    #momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    #kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    #lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    #wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    #lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])
    if run == 'warmup':
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(len(train_loader) * epochs)

    # Create optimizers for train whiten bias stage
    # (in practice we don't need this stage since `whiten_bias_epochs=0` but we keep for ablation)
    model = model_trainbias
    # print([k for k,v in model.named_parameters()])
    first_layer = model._orig_mod[0].weight
    fc_layer = model._orig_mod[-2].weight
    whiten_bias = model._orig_mod[0].bias
    radius = 8.0
    head_radius = 128
    parameters = [
        dict(norm="SpectralConv", norm_kwargs={'steps': 9}, scale=radius, params=[first_layer]),
        dict(norm='SpectralConv', norm_kwargs={'steps': 9}, scale=radius, params=[p for n, p in model.named_parameters() if len(p.shape) == 4 and p.requires_grad and "0.weight" not in n]),
        dict(norm='BiasRMS', scale=radius, params=[p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]),
        dict(norm='Sign', scale=head_radius, params=[fc_layer]),
    ]
    optimizer = Scion(parameters, lr=lr, momentum=momentum)
    optimizer_trainbias = optimizer
    optimizer2_trainbias = Scion(norm='BiasRMS', scale=radius, params=[whiten_bias], lr=lr, momentum=momentum)

    # Create optimizers for frozen whiten bias stage
    model = model_freezebias
    first_layer = model._orig_mod[0].weight
    fc_layer = model._orig_mod[-2].weight
    radius = 8.0
    head_radius = 128
    parameters = [
        dict(norm="SpectralConv", norm_kwargs={'steps': 9}, scale=radius, params=[first_layer]),
        dict(norm='SpectralConv', norm_kwargs={'steps': 9}, scale=radius, params=[p for n, p in model.named_parameters() if len(p.shape) == 4 and p.requires_grad and "0.weight" not in n]),
        dict(norm='BiasRMS', scale=radius, params=[p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]),
        dict(norm='Sign', scale=head_radius, params=[fc_layer]),
    ]
    optimizer = Scion(parameters, lr=lr, momentum=momentum)
    optimizer_freezebias = optimizer

    # Make learning rate schedulers for all 5 optimizers
    def get_lr(step):
        return 1 - step / total_train_steps
    scheduler_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer_trainbias, get_lr)
    scheduler2_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer2_trainbias, get_lr)
    scheduler_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer_freezebias, get_lr)

    # Reinitialize the network from scratch - nothing is reused from previous runs besides the PyTorch compilation
    reinit_net(model_trainbias)
    current_steps = 0
    # for optimizer in [optimizer1_trainbias, optimizer2_trainbias, optimizer22_trainbias, optimizer3_trainbias]:
    #     optimizer.init()

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias._orig_mod[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    # import norm_logging
    # for k,v in norm_logging.log_spectral_norms(model).items():
    #     print(f"{k}: {v}")

    for epoch in range(ceil(epochs)):

        # After training the whiten bias for some epochs, swap in the compiled model with frozen bias
        if epoch == 0:
            model = model_trainbias
            optimizers = [optimizer_trainbias, optimizer2_trainbias]
            schedulers = [scheduler_trainbias, scheduler2_trainbias]
        elif epoch == hyp['opt']['whiten_bias_epochs']:
            model = model_freezebias
            old_optimizers = optimizers
            old_schedulers = schedulers
            optimizers = [optimizer_freezebias]
            schedulers = [scheduler_freezebias]
            model.load_state_dict(model_trainbias.state_dict())
            for i, (opt, sched) in enumerate(zip(optimizers, schedulers)):
                opt.load_state_dict(old_optimizers[i].state_dict())
                sched.load_state_dict(old_schedulers[i].state_dict())

        ####################
        #     Training     #
        ####################

        starter.record()

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
            current_steps += 1
            if current_steps >= total_train_steps:
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    # import norm_logging
    # for k,v in norm_logging.log_spectral_norms(model).items():
    #     print(f"{k}: {v}")

    return tta_val_acc


#######################################################
# Run
#######################################################

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    #for width_factor in [0.5, 1.0, 2.0, 4.0]:
    for width_factor in [1.0]:
        widths = {k: round(width_factor*v) for k,v in hyp['net']['widths'].items()}
        # These two compiled models are first warmed up, and then reinitialized every run. No learned
        # weights are reused between runs. To implement freezing of the whitening-layer bias parameter
        # midway through training, we use two compiled models, one with trainable and the other with
        # frozen whitening bias. This is faster than the naive approach of setting requires_grad=False
        # on the whitening bias midway through training on a single compiled model.
        model_trainbias = make_net(widths)
        model_freezebias = make_net(widths)
        model_freezebias[0].bias.requires_grad = False
        model_trainbias = torch.compile(model_trainbias)#, mode='max-autotune')
        model_freezebias = torch.compile(model_freezebias)#, mode='max-autotune')

        print_columns(logging_columns_list, is_head=True)
        main('warmup', model_trainbias, model_freezebias, lr=0.05, momentum=0.5)
        
        #for log2lr in np.linspace(-9, 0, 10):
        for log2lr in [math.log2(0.05)]:
            momentum = 0.6
            # run = wandb.init(
            #     project="MYPROJECT", 
            #     entity="MYENTITY", 
            #     name=f"v1|scion|transfer|width-factor={width_factor}|log2lr={log2lr}", 
            #     tags=['transfer-v1'],
            #     config={'momentum': momentum, 'log2lr': log2lr, 'method': 'scion', 'width-factor': width_factor}, reinit=True)

            accs = torch.tensor([main(run, model_trainbias, model_freezebias, lr=2**log2lr, momentum=momentum) for run in range(50)])
            print('lr=%d width_facto=%.1f - Mean: %.4f    Std: %.4f' % (log2lr, width_factor, accs.mean(), accs.std()))
            # wandb.log({'test_acc_mean': accs.mean(), 'test_acc_std': accs.std()})
            # run.finish()

