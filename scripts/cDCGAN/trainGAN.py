# library imports
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# ensuring CUDA enbaled GPU is being used

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# create paths to save training results and model

path_num = (len(next(os.walk('generated'))[1]))
path1 = 'models/run' + str(path_num)
path2 = 'generated/run' + str(path_num)

isExist = os.path.exists(path1)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path1)
    print("The new directory is created!")

isExist = os.path.exists(path2)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path2)
    print("The new directory is created!")

path1 = 'models/run' + str(path_num) + '/'
path2 = 'generated/run' + str(path_num) + '/'

# use CUDA device
dev = 'cuda:0'
device = torch.device(dev)

# set parameters & hyper-parameters
image_size = 256
label_dim = 2
G_input_dim = label_dim ** 2
G_output_dim = 1
D_input_dim = 1
D_output_dim = 1
num_filters = [2048, 1024, 512, 256, 128, 64]
learning_rate = 1e-05
betas = (0.5, 0.999)
batch_size = 4
num_epochs = 60
save_dir = path2

# save parameters to text file for future reference
lines = [('image_size: ' + str(image_size)), ('filters: ' + str(num_filters)), ('batch_size: ' + str(batch_size)),
         ('learning_rate: ' + str(learning_rate))]
with open(path2 + 'parameters.txt', 'w') as f:
    for line in lines:
        f.write(str(line))
        f.write('\n')

# initialise instance of transforms for image transformation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))

])

# load in images from 'decompToTrain' folder
data = dsets.ImageFolder('decompToTrain', transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=data,
                                          batch_size=batch_size,
                                          shuffle=True, pin_memory=True)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(input_dim, int(num_filters[i] / 2), kernel_size=4, stride=1,
                                                        padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(int(num_filters[i] / 2)))

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv = torch.nn.ConvTranspose2d(label_dim, int(num_filters[i] / 2), kernel_size=4, stride=1,
                                                        padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal_(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(int(num_filters[i] / 2)))

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2,
                                                  padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(input_dim, int(num_filters[i] / 2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv = torch.nn.Conv2d(label_dim, int(num_filters[i] / 2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)

                # Initializer
                torch.nn.init.normal_(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_conv.bias, 0.0)

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Plot losses
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir=path2, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epoch)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses)) * 1.1)
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'cDCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, label, num_epoch, save=False, save_dir=path2, show=False, fig_size=(200, 200)):
    generator.eval()

    noise = Variable(noise.to(device))
    label = Variable(label.to(device))
    gen_image = generator(noise, label)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
            np.uint8)
        ax.imshow(img, cmap=plt.cm.gray, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch + 1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'cDCGAN_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


"""## Run Model

You can view generated images and loss plots in the 'generated_images' folder.
"""

G = Generator(G_input_dim, label_dim, num_filters, G_output_dim).to(device)
D = Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim).to(device)
# G.cuda()
# D.cuda()

# Loss function
criterion = torch.nn.BCELoss()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

# Fixed noise & label for test
num_test_samples = label_dim ** 2

temp_noise = torch.randn(label_dim, G_input_dim)
fixed_noise = temp_noise
fixed_c = torch.zeros(label_dim, 1)
for i in range(label_dim - 1):
    fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
    temp = torch.ones(label_dim, 1) + i
    fixed_c = torch.cat([fixed_c, temp], 0)

fixed_noise = fixed_noise.view(-1, G_input_dim, 1, 1)
fixed_label = torch.zeros(G_input_dim, label_dim)
fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
fixed_label = fixed_label.view(-1, label_dim, 1, 1)

# label preprocess
onehot = torch.zeros(label_dim, label_dim)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(label_dim, 1), 1).view(
    label_dim, label_dim, 1, 1)
fill = torch.zeros([label_dim, label_dim, image_size, image_size])
for i in range(label_dim):
    fill[i, i, :, :] = 1

# Commented out IPython magic to ensure Python compatibility.
# Training GAN
D_avg_losses = []
G_avg_losses = []

step = 0
for epoch in range(num_epochs):
    D_losses = []
    G_losses = []

    if epoch > 40:
        if epoch % 5 == 0:
            G_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

    # minibatch training
    for i, (images, labels) in enumerate(data_loader):
        try:
            # image data
            mini_batch = images.size()[0]
            x_ = Variable(images.to(device))

            # labels
            y_real_ = Variable(torch.ones(mini_batch).to(device))
            y_fake_ = Variable(torch.zeros(mini_batch).to(device))
            c_fill_ = Variable(fill[labels].to(device))

            # Train discriminator with real data
            D_real_decision = D(x_, c_fill_).squeeze()
            D_real_loss = criterion(D_real_decision, y_real_)

            # Train discriminator with fake data
            z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
            z_ = Variable(z_.to(device))

            c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
            c_onehot_ = Variable(onehot[c_].to(device))
            gen_image = G(z_, c_onehot_)

            c_fill_ = Variable(fill[c_].to(device))
            D_fake_decision = D(gen_image, c_fill_).squeeze()
            D_fake_loss = criterion(D_fake_decision, y_fake_)

            # Back propagation
            D_loss = D_real_loss + D_fake_loss
            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Train generator
            z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
            z_ = Variable(z_.to(device))

            c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
            c_onehot_ = Variable(onehot[c_].to(device))
            gen_image = G(z_, c_onehot_)

            c_fill_ = Variable(fill[c_].to(device))
            D_fake_decision = D(gen_image, c_fill_).squeeze()
            G_loss = criterion(D_fake_decision, y_real_)

            # Back propagation
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # Loss values
            D_losses.append(torch.Tensor.item(D_loss.data))
            G_losses.append(torch.Tensor.item(G_loss.data))

            if epoch % 5 == 0:
                name = path1 + str(epoch) + '.pt'
                torch.save(G.state_dict(), name)

            # name = 'models/gen' + str(epoch) + '.pt'
            # torch.save(G.state_dict(), name)

            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(data_loader), torch.Tensor.item(D_loss.data),
                     torch.Tensor.item(G_loss.data)))
        except ValueError:
            pass
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # Avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    plot_loss(D_avg_losses, G_avg_losses, epoch, save=True, save_dir=save_dir)

    # Show result for fixed noise
    plot_result(G, fixed_noise, fixed_label, epoch, save=True, save_dir=save_dir)

# Make gif
loss_plots = []
gen_image_plots = []
for epoch in range(num_epochs):
    # Plot for generating gif
    save_fn1 = save_dir + 'cDCGAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
    loss_plots.append(imageio.v2.imread(save_fn1))

    save_fn2 = save_dir + 'cDCGAN_epoch_{:d}'.format(epoch + 1) + '.png'
    gen_image_plots.append(imageio.v2.imread(save_fn2))

imageio.mimsave(save_dir + 'cDCGAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=5)
imageio.mimsave(save_dir + 'cDCGAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)
