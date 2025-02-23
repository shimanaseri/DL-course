#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')


# In[ ]:


import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16

            # img: 32x32 after this block
            nn.ConvTranspose2d(
                features_g * 4,
                features_g * 4,  # Keeping the number of channels same for PixelShuffle
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),

            # PixelShuffle layer
            nn.PixelShuffle(upscale_factor=2),  # img: 64x64 after PixelShuffle

            # Final Convolution to get to the right number of channels
            nn.Conv2d(
                features_g,  # Adjusted due to PixelShuffle
                channels_img,
                kernel_size=3,  # Using a 3x3 kernel for final adjustments
                stride=1,
                padding=1
            ),
            nn.Tanh(),  # Output: N x channels_img x 64 x 64
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)




def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 1, 64, 64  # Set in_channels to 1 for MNIST
    noise_dim = 64
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")



if __name__ == "__main__":
    test()


# # Simple GAN

# In[ ]:


import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torchvision.datasets as dss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 16
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 64
NUM_EPOCHS = 10
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = dss.MNIST(
    root="dataset/", train=True, transform=transforms, download=True
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs_GAN/real")
writer_fake = SummaryWriter(f"logs_GAN/fake")
writer_losses = SummaryWriter(f"logs_GAN/losses")



# In[ ]:


step = 0
gen.train()
disc.train()

real_images = []
generated_images = []
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)


        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        writer_losses.add_scalar("Loss/Discriminator", loss_disc.item(), global_step=step)
        writer_losses.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)

        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )


            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
writer_losses.close()


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir /content/logs_GAN')


# # W-GAN

# In[11]:


import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 256
NUM_EPOCHS = 3
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 2
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

# for tensorboard plotting
fixed_noise = torch.randn(28, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs_WGAN/real")
writer_fake = SummaryWriter(f"logs_WGAN/fake")
writer_losses = SummaryWriter(f"logs_WGAN/losses")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # clip critic weights between -0.01, 0.01
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        writer_losses.add_scalar("Loss/Critic", loss_critic.item(), global_step=step)
        writer_losses.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)

        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    data[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)


            step += 1
            gen.train()
            critic.train()
writer_losses.close()


# In[13]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir /content/logs_WGAN')


# # SSGAN

# In[ ]:


import torch
import torch.nn as nn


import torch
import torch.nn as nn

# Residual Block definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, UpS=False, DnS=False):
        super(ResidualBlock, self).__init__()
        self.UpS = UpS
        self.DnS = DnS

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True) if not DnS else nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest') if UpS else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest') if UpS else nn.Identity()
        )

    def forward(self, x):
        return x + self.block1(x)

# Generator definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(128, 256 * 4 * 4)
        self.residual1 = ResidualBlock(256, 256, UpS=True)
        self.residual2 = ResidualBlock(256, 256, UpS=True)
        self.residual3 = ResidualBlock(256, 256, UpS=True)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlock(1, 128, DnS=True)
        self.residual2 = ResidualBlock(128, 128, DnS=True)
        self.residual3 = ResidualBlock(128, 128)
        self.residual4 = ResidualBlock(128, 128)
        self.linear = nn.Linear(128, 1)
        self.linear_class = nn.Linear(128, 4)

    def forward(self, x):
        x1 = self.residual1(x)
        x2 = self.residual2(x1)
        x3 = self.residual3(x2)
        x4 = self.residual4(x3)

        x1_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)(x1)
        x2_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)(x2)
        x2_conv1x1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)(x2_avgpool)
        x = x1_avgpool + x2_conv1x1
        x = self.linear(x.view(x.size(0), -1))
        x_class = self.linear_class(x.view(x.size(0), -1))

        return x, x_class

# # Instantiate the generator and discriminator
# generator = Generator()
# discriminator = Discriminator()

# # Print the models
# print(generator)
# print(discriminator)



def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 1, 64, 64  # Set in_channels to 1 for MNIST
    noise_dim = 64
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator()
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator()
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")



if __name__ == "__main__":
    test()

