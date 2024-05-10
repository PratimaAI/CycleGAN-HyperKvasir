```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image

# Inspired by https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the horse2zebra images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))
```


```python
class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class:
    Performs two convolutions and an instance normalization, the input is added
    to this output to form the residual block output.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ResidualBlock:
        Given an image tensor, completes a residual block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x
```


```python
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to upsample,
        with an optional instance norm
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator -
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x
```


```python
class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator:
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)
```


```python
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn
```


```python
import torch.nn.functional as F

adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

n_epochs = 20
dim_A = 3
dim_B = 3
display_step = 200
batch_size = 1
lr = 0.0002
load_shape = 286
target_shape = 256
device = 'cuda'
```


```python
transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

import torchvision
dataset = ImageDataset("Horse2Zebra_Dataset", transform=transform)
```


```python
gen_AB = Generator(dim_A, dim_B)
gen_BA = Generator(dim_B, dim_A)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Discriminator(dim_A)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Discriminator(dim_B)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

# Feel free to change pretrained to False if you're training the model from scratch
pretrained = False
if pretrained:
    pre_dict = torch.load('cycleGAN_100000.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])
    gen_BA.load_state_dict(pre_dict['gen_BA'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    disc_A.load_state_dict(pre_dict['disc_A'])
    disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
    disc_B.load_state_dict(pre_dict['disc_B'])
    disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
else:
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)
```


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator
            predictions and the target labels and returns a adversarial
            loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    pred_fake = disc_X(fake_X)
    target = torch.zeros_like(pred_fake)
    loss1 = adv_criterion(pred_fake, target)

    pred_real = disc_X(real_X)
    target = torch.ones_like(pred_real)
    loss2 = adv_criterion(pred_real, target)
    disc_loss = (loss1 + loss2) / 2
    #### END CODE HERE ####
    return disc_loss
```


```python
# UNIT TEST
test_disc_X = lambda x: x * 97
test_real_X = torch.tensor(83.)
test_fake_X = torch.tensor(89.)
test_adv_criterion = lambda x, y: x * 79 + y * 73
assert torch.abs((get_disc_loss(test_real_X, test_fake_X, test_disc_X, test_adv_criterion)) - 659054.5000) < 1e-6
test_disc_X = lambda x: x.mean(0, keepdim=True)
test_adv_criterion = torch.nn.BCEWithLogitsLoss()
test_input = torch.ones(20, 10)
# If this runs, it's a pass - checks that the shapes are treated correctly
get_disc_loss(test_input, test_input, test_disc_X, test_adv_criterion)
print("Success!")
```

    Success!



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_adversarial_loss
def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator
                  predictions and the target labels and returns a adversarial
                  loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    fake_Y = gen_XY(real_X)
    pred_fake = disc_Y(fake_Y)
    target = torch.ones_like(pred_fake)
    adversarial_loss = adv_criterion(pred_fake, target)

    #### END CODE HERE ####
    return adversarial_loss, fake_Y
```


```python
# UNIT TEST
test_disc_Y = lambda x: x * 97
test_real_X = torch.tensor(83.)
test_gen_XY = lambda x: x * 89
test_adv_criterion = lambda x, y: x * 79 + y * 73
test_res = get_gen_adversarial_loss(test_real_X, test_disc_Y, test_gen_XY, test_adv_criterion)
assert torch.abs(test_res[0] - 56606652) < 1e-6
assert torch.abs(test_res[1] - 7387) < 1e-6
test_disc_Y = lambda x: x.mean(0, keepdim=True)
test_adv_criterion = torch.nn.BCEWithLogitsLoss()
test_input = torch.ones(20, 10)
# If this runs, it's a pass - checks that the shapes are treated correctly
get_gen_adversarial_loss(test_input, test_disc_Y, test_gen_XY, test_adv_criterion)
print("Success!")
```

    Success!



```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_identity_loss
def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity
                        loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)
    #### END CODE HERE ####
    return identity_loss, identity_X
```


```python
# UNIT TEST
test_real_X = torch.tensor(83.)
test_gen_YX = lambda x: x * 89
test_identity_criterion = lambda x, y: (x + y) * 73
test_res = get_identity_loss(test_real_X, test_gen_YX, test_identity_criterion)
assert torch.abs(test_res[0] - 545310) < 1e-6
assert torch.abs(test_res[1] - 7387) < 1e-6
print("Success!")
```

    Success!



```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_cycle_consistency_loss
def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    #### END CODE HERE ####
    return cycle_loss, cycle_X
```


```python
# UNIT TEST
test_real_X = torch.tensor(83.)
test_fake_Y = torch.tensor(97.)
test_gen_YX = lambda x: x * 89
test_cycle_criterion = lambda x, y: (x + y) * 73
test_res = get_cycle_consistency_loss(test_real_X, test_fake_Y, test_gen_YX, test_cycle_criterion)
assert torch.abs(test_res[1] - 8633) < 1e-6
assert torch.abs(test_res[0] - 636268) < 1e-6
print("Success!")
```

    Success!



```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator
            predictions and the true labels and returns a adversarial
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''
    # Hint 1: Make sure you include both directions - you can think of the generators as collaborating
    # Hint 2: Don't forget to use the lambdas for the identity loss and cycle loss!
    #### START CODE HERE ####
    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion) # G : A -> B
    adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_A, identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B, identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)
    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_A, cycle_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_B, cycle_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    # Total loss
    gen_loss = adv_loss_AB + adv_loss_BA + lambda_identity * (identity_loss_A + identity_loss_B) + lambda_cycle * (cycle_loss_A + cycle_loss_B)
    #### END CODE HERE ####
    return gen_loss, fake_A, fake_B
```


```python
# UNIT TEST
test_real_A = torch.tensor(97)
test_real_B = torch.tensor(89)
test_gen_AB = lambda x: x * 83
test_gen_BA = lambda x: x * 79
test_disc_A = lambda x: x * 47
test_disc_B = lambda x: x * 43
test_adv_criterion = lambda x, y: x * 73 + y * 71
test_recon_criterion = lambda x, y: (x + y) * 61
test_lambda_identity = 59
test_lambda_cycle = 53
test_res = get_gen_loss(
    test_real_A,
    test_real_B,
    test_gen_AB,
    test_gen_BA,
    test_disc_A,
    test_disc_B,
    test_adv_criterion,
    test_recon_criterion,
    test_recon_criterion,
    test_lambda_identity,
    test_lambda_cycle)
assert test_res[0].item() == 4047804560
assert test_res[1].item() == 7031
assert test_res[2].item() == 8051
print("Success!")
```

    Success!



```python
from skimage import color
import numpy as np
plt.rcParams["figure.figsize"] = (10, 10)


def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        # for image, _ in tqdm(dataloader):
        for real_A, real_B in tqdm(dataloader):
            # image_width = image.shape[3]
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A
            real_B = real_B

            ### Update discriminator A ###
            disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True) # Update gradients
            disc_A_opt.step() # Update optimizer

            ### Update discriminator B ###
            disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True) # Update gradients
            disc_B_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"cycleGAN_{cur_step}.pth")
            cur_step += 1
train(save_model=True)
```


      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 0: Step 0: Generator (U-Net) loss: 0.06467757701873779, Discriminator loss: 0.0029837241768836975




![png](output_18_2.png)





![png](output_18_3.png)



    Epoch 0: Step 200: Generator (U-Net) loss: 6.893248019218444, Discriminator loss: 0.2619549982249736




![png](output_18_5.png)





![png](output_18_6.png)



    Epoch 0: Step 400: Generator (U-Net) loss: 6.0488439726829535, Discriminator loss: 0.22916086256504062




![png](output_18_8.png)





![png](output_18_9.png)



    Epoch 0: Step 600: Generator (U-Net) loss: 5.890418623685838, Discriminator loss: 0.23365519396960732




![png](output_18_11.png)





![png](output_18_12.png)



    Epoch 0: Step 800: Generator (U-Net) loss: 5.656403275728228, Discriminator loss: 0.22695812303572896




![png](output_18_14.png)





![png](output_18_15.png)



    Epoch 0: Step 1000: Generator (U-Net) loss: 5.5149971044063575, Discriminator loss: 0.22142785593867273




![png](output_18_17.png)





![png](output_18_18.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 1: Step 1200: Generator (U-Net) loss: 5.496146886348724, Discriminator loss: 0.2248021974042059




![png](output_18_21.png)





![png](output_18_22.png)



    Epoch 1: Step 1400: Generator (U-Net) loss: 5.3431181931495635, Discriminator loss: 0.2233133454620838




![png](output_18_24.png)





![png](output_18_25.png)



    Epoch 1: Step 1600: Generator (U-Net) loss: 5.145296689271923, Discriminator loss: 0.22531796406954532




![png](output_18_27.png)





![png](output_18_28.png)



    Epoch 1: Step 1800: Generator (U-Net) loss: 5.52548555970192, Discriminator loss: 0.22576146446168435




![png](output_18_30.png)





![png](output_18_31.png)



    Epoch 1: Step 2000: Generator (U-Net) loss: 4.905784214735029, Discriminator loss: 0.23360561374574898




![png](output_18_33.png)





![png](output_18_34.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 2: Step 2200: Generator (U-Net) loss: 5.016619963645935, Discriminator loss: 0.22253344975411893




![png](output_18_37.png)





![png](output_18_38.png)



    Epoch 2: Step 2400: Generator (U-Net) loss: 4.985986202955244, Discriminator loss: 0.2257054983451962




![png](output_18_40.png)





![png](output_18_41.png)



    Epoch 2: Step 2600: Generator (U-Net) loss: 4.928472909927369, Discriminator loss: 0.22582008417695779




![png](output_18_43.png)





![png](output_18_44.png)



    Epoch 2: Step 2800: Generator (U-Net) loss: 4.806403677463531, Discriminator loss: 0.22585417803376925




![png](output_18_46.png)





![png](output_18_47.png)



    Epoch 2: Step 3000: Generator (U-Net) loss: 4.8555942106246945, Discriminator loss: 0.22701797846704733




![png](output_18_49.png)





![png](output_18_50.png)



    Epoch 2: Step 3200: Generator (U-Net) loss: 4.782143088579178, Discriminator loss: 0.22431446325033907




![png](output_18_52.png)





![png](output_18_53.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 3: Step 3400: Generator (U-Net) loss: 4.716583569049835, Discriminator loss: 0.2276625248789787




![png](output_18_56.png)





![png](output_18_57.png)



    Epoch 3: Step 3600: Generator (U-Net) loss: 4.950330408811567, Discriminator loss: 0.22171373385936025




![png](output_18_59.png)





![png](output_18_60.png)



    Epoch 3: Step 3800: Generator (U-Net) loss: 4.774811098575593, Discriminator loss: 0.22534988284111032




![png](output_18_62.png)





![png](output_18_63.png)



    Epoch 3: Step 4000: Generator (U-Net) loss: 4.843809188604357, Discriminator loss: 0.21615444187074892




![png](output_18_65.png)





![png](output_18_66.png)



    Epoch 3: Step 4200: Generator (U-Net) loss: 4.596182411909105, Discriminator loss: 0.22512697994709016




![png](output_18_68.png)





![png](output_18_69.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 4: Step 4400: Generator (U-Net) loss: 4.547185761928557, Discriminator loss: 0.23030764028429987




![png](output_18_72.png)





![png](output_18_73.png)



    Epoch 4: Step 4600: Generator (U-Net) loss: 4.625526288747786, Discriminator loss: 0.22357677757740022




![png](output_18_75.png)





![png](output_18_76.png)



    Epoch 4: Step 4800: Generator (U-Net) loss: 4.612153434753418, Discriminator loss: 0.2201051012054087




![png](output_18_78.png)





![png](output_18_79.png)



    Epoch 4: Step 5000: Generator (U-Net) loss: 4.596741224527363, Discriminator loss: 0.22529190257191653




![png](output_18_81.png)





![png](output_18_82.png)



    Epoch 4: Step 5200: Generator (U-Net) loss: 4.64387490987778, Discriminator loss: 0.2138954669609666




![png](output_18_84.png)





![png](output_18_85.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 5: Step 5400: Generator (U-Net) loss: 4.502754776477812, Discriminator loss: 0.22488026246428483




![png](output_18_88.png)





![png](output_18_89.png)



    Epoch 5: Step 5600: Generator (U-Net) loss: 4.666863645315174, Discriminator loss: 0.2182574951648713




![png](output_18_91.png)





![png](output_18_92.png)



    Epoch 5: Step 5800: Generator (U-Net) loss: 4.690556684732436, Discriminator loss: 0.21897840786725284




![png](output_18_94.png)





![png](output_18_95.png)



    Epoch 5: Step 6000: Generator (U-Net) loss: 4.389385305643086, Discriminator loss: 0.21293415911495694




![png](output_18_97.png)





![png](output_18_98.png)



    Epoch 5: Step 6200: Generator (U-Net) loss: 4.577961940765381, Discriminator loss: 0.21841237898916022




![png](output_18_100.png)





![png](output_18_101.png)



    Epoch 5: Step 6400: Generator (U-Net) loss: 4.57608028292656, Discriminator loss: 0.21281103312969216




![png](output_18_103.png)





![png](output_18_104.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 6: Step 6600: Generator (U-Net) loss: 4.681439833641051, Discriminator loss: 0.21432575382292263




![png](output_18_107.png)





![png](output_18_108.png)



    Epoch 6: Step 6800: Generator (U-Net) loss: 4.520278652906417, Discriminator loss: 0.2106325830891729




![png](output_18_110.png)





![png](output_18_111.png)



    Epoch 6: Step 7000: Generator (U-Net) loss: 4.404565739631652, Discriminator loss: 0.21685904663056127




![png](output_18_113.png)





![png](output_18_114.png)



    Epoch 6: Step 7200: Generator (U-Net) loss: 4.4815626657009116, Discriminator loss: 0.20918500876054175




![png](output_18_116.png)





![png](output_18_117.png)



    Epoch 6: Step 7400: Generator (U-Net) loss: 4.464911108016967, Discriminator loss: 0.21225156189873823




![png](output_18_119.png)





![png](output_18_120.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 7: Step 7600: Generator (U-Net) loss: 4.285716462135318, Discriminator loss: 0.21263580147176983




![png](output_18_123.png)





![png](output_18_124.png)



    Epoch 7: Step 7800: Generator (U-Net) loss: 4.476246684789659, Discriminator loss: 0.2110152145475149




![png](output_18_126.png)





![png](output_18_127.png)



    Epoch 7: Step 8000: Generator (U-Net) loss: 4.395134657621382, Discriminator loss: 0.20756121698766933




![png](output_18_129.png)





![png](output_18_130.png)



    Epoch 7: Step 8200: Generator (U-Net) loss: 4.4040322172641755, Discriminator loss: 0.20630884420126683




![png](output_18_132.png)





![png](output_18_133.png)



    Epoch 7: Step 8400: Generator (U-Net) loss: 4.400272458791732, Discriminator loss: 0.20959923647344106




![png](output_18_135.png)





![png](output_18_136.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 8: Step 8600: Generator (U-Net) loss: 4.374053440093993, Discriminator loss: 0.20313889637589463




![png](output_18_139.png)





![png](output_18_140.png)



    Epoch 8: Step 8800: Generator (U-Net) loss: 4.373019713163378, Discriminator loss: 0.21182742692530165




![png](output_18_142.png)





![png](output_18_143.png)



    Epoch 8: Step 9000: Generator (U-Net) loss: 4.341354665756225, Discriminator loss: 0.2136691066622735




![png](output_18_145.png)





![png](output_18_146.png)



    Epoch 8: Step 9200: Generator (U-Net) loss: 4.510362569093701, Discriminator loss: 0.20832215370610357




![png](output_18_148.png)





![png](output_18_149.png)



    Epoch 8: Step 9400: Generator (U-Net) loss: 4.414220896959307, Discriminator loss: 0.20655924160033465




![png](output_18_151.png)





![png](output_18_152.png)



    Epoch 8: Step 9600: Generator (U-Net) loss: 4.24997017621994, Discriminator loss: 0.21210694402456287




![png](output_18_154.png)





![png](output_18_155.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 9: Step 9800: Generator (U-Net) loss: 4.359142976999285, Discriminator loss: 0.20468582950532438




![png](output_18_158.png)





![png](output_18_159.png)



    Epoch 9: Step 10000: Generator (U-Net) loss: 4.371653797626494, Discriminator loss: 0.20510210670530804




![png](output_18_161.png)





![png](output_18_162.png)



    Epoch 9: Step 10200: Generator (U-Net) loss: 4.329976818561554, Discriminator loss: 0.20577879995107634




![png](output_18_164.png)





![png](output_18_165.png)



    Epoch 9: Step 10400: Generator (U-Net) loss: 4.221572748422622, Discriminator loss: 0.20771416824311015




![png](output_18_167.png)





![png](output_18_168.png)



    Epoch 9: Step 10600: Generator (U-Net) loss: 4.390147535800936, Discriminator loss: 0.21277888838201756




![png](output_18_170.png)





![png](output_18_171.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 10: Step 10800: Generator (U-Net) loss: 4.206376057863234, Discriminator loss: 0.19826772484928373




![png](output_18_174.png)





![png](output_18_175.png)



    Epoch 10: Step 11000: Generator (U-Net) loss: 4.360678057670594, Discriminator loss: 0.2112926612794399




![png](output_18_177.png)





![png](output_18_178.png)



    Epoch 10: Step 11200: Generator (U-Net) loss: 4.172277950048447, Discriminator loss: 0.20822429902851583




![png](output_18_180.png)





![png](output_18_181.png)



    Epoch 10: Step 11400: Generator (U-Net) loss: 4.356290787458422, Discriminator loss: 0.19386100342497234




![png](output_18_183.png)





![png](output_18_184.png)



    Epoch 10: Step 11600: Generator (U-Net) loss: 4.260333688259124, Discriminator loss: 0.20174232814460993




![png](output_18_186.png)





![png](output_18_187.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 11: Step 11800: Generator (U-Net) loss: 4.238070229291917, Discriminator loss: 0.19980014067143204




![png](output_18_190.png)





![png](output_18_191.png)



    Epoch 11: Step 12000: Generator (U-Net) loss: 4.213552827835084, Discriminator loss: 0.20090193066746




![png](output_18_193.png)





![png](output_18_194.png)



    Epoch 11: Step 12200: Generator (U-Net) loss: 4.200289472341539, Discriminator loss: 0.20598574507981532




![png](output_18_196.png)





![png](output_18_197.png)



    Epoch 11: Step 12400: Generator (U-Net) loss: 4.144212930202484, Discriminator loss: 0.19263982899487014




![png](output_18_199.png)





![png](output_18_200.png)



    Epoch 11: Step 12600: Generator (U-Net) loss: 4.2710945677757275, Discriminator loss: 0.19861816233024004




![png](output_18_202.png)





![png](output_18_203.png)



    Epoch 11: Step 12800: Generator (U-Net) loss: 4.257507234811783, Discriminator loss: 0.20400024332106115




![png](output_18_205.png)





![png](output_18_206.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 12: Step 13000: Generator (U-Net) loss: 4.227089745998381, Discriminator loss: 0.19463741745799773




![png](output_18_209.png)





![png](output_18_210.png)



    Epoch 12: Step 13200: Generator (U-Net) loss: 4.128166199922562, Discriminator loss: 0.19121617965400226




![png](output_18_212.png)





![png](output_18_213.png)



    Epoch 12: Step 13400: Generator (U-Net) loss: 4.1609522664546965, Discriminator loss: 0.2034275611303747




![png](output_18_215.png)





![png](output_18_216.png)



    Epoch 12: Step 13600: Generator (U-Net) loss: 4.145009763240814, Discriminator loss: 0.20408420873805871




![png](output_18_218.png)





![png](output_18_219.png)



    Epoch 12: Step 13800: Generator (U-Net) loss: 4.077298326492313, Discriminator loss: 0.19250801360234612




![png](output_18_221.png)





![png](output_18_222.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 13: Step 14000: Generator (U-Net) loss: 4.248200899362564, Discriminator loss: 0.19323008041828865




![png](output_18_225.png)





![png](output_18_226.png)



    Epoch 13: Step 14200: Generator (U-Net) loss: 4.259568943977358, Discriminator loss: 0.19118582751601937




![png](output_18_228.png)





![png](output_18_229.png)



    Epoch 13: Step 14400: Generator (U-Net) loss: 4.120617583990098, Discriminator loss: 0.18799630064517264




![png](output_18_231.png)





![png](output_18_232.png)



    Epoch 13: Step 14600: Generator (U-Net) loss: 4.114803003072739, Discriminator loss: 0.20203654911369082




![png](output_18_234.png)





![png](output_18_235.png)



    Epoch 13: Step 14800: Generator (U-Net) loss: 4.215578159093858, Discriminator loss: 0.19574928697198626




![png](output_18_237.png)





![png](output_18_238.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 14: Step 15000: Generator (U-Net) loss: 4.2560546481609345, Discriminator loss: 0.19273018108680853




![png](output_18_241.png)





![png](output_18_242.png)



    Epoch 14: Step 15200: Generator (U-Net) loss: 4.189760303497315, Discriminator loss: 0.1927676005661486




![png](output_18_244.png)





![png](output_18_245.png)



    Epoch 14: Step 15400: Generator (U-Net) loss: 4.166693423986437, Discriminator loss: 0.18771080488339073




![png](output_18_247.png)





![png](output_18_248.png)



    Epoch 14: Step 15600: Generator (U-Net) loss: 4.2762315356731415, Discriminator loss: 0.1884099473431707




![png](output_18_250.png)





![png](output_18_251.png)



    Epoch 14: Step 15800: Generator (U-Net) loss: 3.969930361509323, Discriminator loss: 0.191533070821315




![png](output_18_253.png)





![png](output_18_254.png)



    Epoch 14: Step 16000: Generator (U-Net) loss: 4.211000576019288, Discriminator loss: 0.19238764217123397




![png](output_18_256.png)





![png](output_18_257.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 15: Step 16200: Generator (U-Net) loss: 4.269913769960403, Discriminator loss: 0.18832037383690475




![png](output_18_260.png)





![png](output_18_261.png)



    Epoch 15: Step 16400: Generator (U-Net) loss: 4.2147509407997115, Discriminator loss: 0.1785095347277819




![png](output_18_263.png)





![png](output_18_264.png)



    Epoch 15: Step 16600: Generator (U-Net) loss: 4.122338333129883, Discriminator loss: 0.1969985600933433




![png](output_18_266.png)





![png](output_18_267.png)



    Epoch 15: Step 16800: Generator (U-Net) loss: 4.095800589323043, Discriminator loss: 0.18610601503401977




![png](output_18_269.png)





![png](output_18_270.png)



    Epoch 15: Step 17000: Generator (U-Net) loss: 4.114018353223797, Discriminator loss: 0.18316973838955156




![png](output_18_272.png)





![png](output_18_273.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 16: Step 17200: Generator (U-Net) loss: 4.110617240667344, Discriminator loss: 0.19605779748409988




![png](output_18_276.png)





![png](output_18_277.png)



    Epoch 16: Step 17400: Generator (U-Net) loss: 4.192730485200883, Discriminator loss: 0.1919423674419523




![png](output_18_279.png)





![png](output_18_280.png)



    Epoch 16: Step 17600: Generator (U-Net) loss: 4.172758113145828, Discriminator loss: 0.19398533329367643




![png](output_18_282.png)





![png](output_18_283.png)



    Epoch 16: Step 17800: Generator (U-Net) loss: 4.097178052663805, Discriminator loss: 0.18039792643859978




![png](output_18_285.png)





![png](output_18_286.png)



    Epoch 16: Step 18000: Generator (U-Net) loss: 4.0355769956111915, Discriminator loss: 0.19204914957284938




![png](output_18_288.png)





![png](output_18_289.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 17: Step 18200: Generator (U-Net) loss: 4.040406547784805, Discriminator loss: 0.18240215612575403




![png](output_18_292.png)





![png](output_18_293.png)



    Epoch 17: Step 18400: Generator (U-Net) loss: 4.117928991317748, Discriminator loss: 0.18352279970422378




![png](output_18_295.png)





![png](output_18_296.png)



    Epoch 17: Step 18600: Generator (U-Net) loss: 4.072173155546188, Discriminator loss: 0.18532326530665166




![png](output_18_298.png)





![png](output_18_299.png)



    Epoch 17: Step 18800: Generator (U-Net) loss: 4.034822124242783, Discriminator loss: 0.1796146794129164




![png](output_18_301.png)





![png](output_18_302.png)



    Epoch 17: Step 19000: Generator (U-Net) loss: 4.108735476732254, Discriminator loss: 0.1858350571990013




![png](output_18_304.png)





![png](output_18_305.png)



    Epoch 17: Step 19200: Generator (U-Net) loss: 4.127135486602785, Discriminator loss: 0.18528029136359692




![png](output_18_307.png)





![png](output_18_308.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 18: Step 19400: Generator (U-Net) loss: 4.088622539043426, Discriminator loss: 0.1926313463598489




![png](output_18_311.png)





![png](output_18_312.png)



    Epoch 18: Step 19600: Generator (U-Net) loss: 3.985123002529145, Discriminator loss: 0.18516561485826966




![png](output_18_314.png)





![png](output_18_315.png)



    Epoch 18: Step 19800: Generator (U-Net) loss: 4.120123189687731, Discriminator loss: 0.18527003079652787




![png](output_18_317.png)





![png](output_18_318.png)



    Epoch 18: Step 20000: Generator (U-Net) loss: 4.058066431283952, Discriminator loss: 0.18607310593128187




![png](output_18_320.png)





![png](output_18_321.png)



    Epoch 18: Step 20200: Generator (U-Net) loss: 4.069167890548708, Discriminator loss: 0.18131198700517415




![png](output_18_323.png)





![png](output_18_324.png)




      0%|          | 0/1067 [00:00<?, ?it/s]


    Epoch 19: Step 20400: Generator (U-Net) loss: 4.07637888908386, Discriminator loss: 0.18210431849583966




![png](output_18_327.png)





![png](output_18_328.png)



    Epoch 19: Step 20600: Generator (U-Net) loss: 4.053164811134338, Discriminator loss: 0.17355104204267272




![png](output_18_330.png)





![png](output_18_331.png)



    Epoch 19: Step 20800: Generator (U-Net) loss: 3.9454812312126166, Discriminator loss: 0.1917514549940823




![png](output_18_333.png)





![png](output_18_334.png)



    Epoch 19: Step 21000: Generator (U-Net) loss: 4.01539737582207, Discriminator loss: 0.1802955007553101




![png](output_18_336.png)





![png](output_18_337.png)



    Epoch 19: Step 21200: Generator (U-Net) loss: 4.068592245578765, Discriminator loss: 0.17425009135156863




![png](output_18_339.png)





![png](output_18_340.png)




```python
print("Generator Architecture:")
print(gen_AB)

print("\nDiscriminator Architecture for Domain A:")
print(disc_A)

print("\nDiscriminator Architecture for Domain B:")
print(disc_B)

```

    Generator Architecture:



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 2
          1 print("Generator Architecture:")
    ----> 2 print(gen_AB)
          4 print("\nDiscriminator Architecture for Domain A:")
          5 print(disc_A)


    NameError: name 'gen_AB' is not defined



```python

```
