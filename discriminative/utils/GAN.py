import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

n_epochs = 50
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 100
num_classes = 2
img_size = 28
channels = 1
sample_interval = 400
threshold = 0.99
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



class VGR():
    def __init__(self, task_id):
        self.task_id = task_id
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.auxiliary_loss.cuda()

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))


    def train(self, x_train, y_train):
        N = x_train.shape[0]
        x_train -= 0.5
        x_train /= 0.5
        for epoch in range(n_epochs):

            total_batch = int(np.ceil(N * 1.0 / batch_size))
            perm_inds = np.arange(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = device)
                batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = device)
                batch_x = batch_x.reshape(-1,img_size,img_size).unsqueeze(1)
                bsize = batch_x.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(bsize, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(bsize, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(batch_x.type(FloatTensor))
                labels = Variable(batch_y.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (bsize, latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                validity, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.discriminator(real_imgs)
                d_real_loss =  (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = real_aux.data.cpu().numpy()
                gt = labels.data.cpu().numpy()
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.optimizer_D.step()

            print ("Epoch {}/{},  Discriminator loss: {}, acc: {}%, Generator loss: {}".format(epoch, n_epochs, d_loss.item(), 100 * d_acc, g_loss.item()))


        self.generator.cpu()
        self.discriminator.cpu()
        self.adversarial_loss.cpu()
        self.auxiliary_loss.cpu()



    def generate_samples(self, no_samples, task_id, current_nb = 0):
         # Sample noise and labels as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (no_samples, latent_dim))))
        # Generate a batch of images
        gen_imgs = self.generator(z)
        _,labels = self.discriminator(gen_imgs)
        kept_indices = torch.nonzero(torch.where(torch.max(labels,1)[0] < threshold, torch.zeros(labels.shape[0]), torch.ones(labels.shape[0]))).squeeze(1)
        print(kept_indices.shape[0])
        labels = labels.index_select(0,kept_indices)
        gen_imgs = gen_imgs.index_select(0,kept_indices)
        if current_nb == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % task_id, nrow=5, normalize=True)
        gen_imgs = gen_imgs.squeeze(1).reshape(kept_indices.shape[0],784)
        labels = labels.argmax(1).type(torch.FloatTensor)
        gen_imgs = gen_imgs.data.cpu()
        labels = labels.data.cpu()


        new_current_nb = current_nb + kept_indices.shape[0]
        if(new_current_nb < no_samples):
            new_gen_imgs, new_labels = self.generate_samples(no_samples,task_id, new_current_nb)
            gen_imgs = np.vstack((gen_imgs,new_gen_imgs))
            labels = np.hstack((labels,new_labels))

        return gen_imgs*0.5+0.5, labels

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(512, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(512, num_classes),
                                        nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


