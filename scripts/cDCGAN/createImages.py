import os
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from numpy.random import randn, randint
from numpy import asarray
from PIL import Image
import pickle
import statistics
import hashlib

dev = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
device = torch.device(dev)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


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


image_size = 256
label_dim = 2
G_input_dim = label_dim ** 2
G_output_dim = 1
D_input_dim = 1
D_output_dim = 1

num_filters = [4096,2048, 1024, 512, 256, 128]

G = Generator(G_input_dim, label_dim, num_filters, G_output_dim)

model = Generator(G_input_dim, label_dim, num_filters, G_output_dim).cuda()

model.load_state_dict(torch.load('models/run25/75.pt'))

# inception = pickle.load(open('ZHANG_PNEU_model.sav', 'rb'))
inception = tf.keras.models.load_model('ZHANG_PNEU_model.h5')

labels = {0: 'NORMAL_CLUSTER_0', 1: 'NORMAL_CLUSTER_1'}
numbers = {0: 100, 1: 100}

fake_path = 'fake/'
folder_count = len(os.listdir(fake_path))
main_path = 'fake/' + 'created' + str(folder_count)
isExist = os.path.exists(main_path)
if not isExist:
    os.makedirs(main_path)
    print(f"Created: {main_path}")
for x in range(label_dim):
    created = 0
    toGen = x
    path = main_path + '/' + labels[toGen]
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    class_to_gen = labels[toGen]
    class_path = 'decompToTrainVal/' + class_to_gen
    PROBABILITIES = []
    for i in range(0, 100):
        try:
            path = os.listdir(class_path)[i]
            img_path = class_path + '/' + path
            img = Image.open(img_path)
            img = img.resize((150, 150))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            x = np.asarray(img)
            x = np.expand_dims(x, axis=0)
            preds = inception.predict(x, verbose=0)
            label = np.argmax(preds[0])
            probability = preds[0][label]
            PROBABILITIES.append(probability)
        except IndexError:
            pass

    average_probability = (statistics.mean(PROBABILITIES))
    while created < (numbers[toGen]):
        temp_noise = torch.randn(label_dim, label_dim ** 2)
        fixed_noise = temp_noise
        fixed_c = torch.zeros(label_dim, 1)
        fixed_c[0] = toGen
        for i in range(label_dim - 1):
            fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
            temp = torch.ones(label_dim, 1) + i
            fixed_c = torch.cat([fixed_c, temp], 0)

        fixed_noise = fixed_noise.view(-1, G_input_dim, 1, 1)
        fixed_label = torch.zeros(G_input_dim, label_dim)
        fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
        fixed_label = fixed_label.view(-1, label_dim, 1, 1)

        noise = Variable(fixed_noise.cuda())
        label = Variable(fixed_label.cuda())
        gen_image = model(noise, label).detach()
        gen_image = denorm(gen_image)
        gen_image = gen_image[0]
        gen_image = torchvision.transforms.ToPILImage()(gen_image)
        img = gen_image
        img = img.resize((150, 150))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x = np.asarray(img)
        x = np.expand_dims(x, axis=0)
        preds = inception.predict(x, verbose=0)
        label = np.argmax(preds[0])
        probability = preds[0][label]
        print(probability)
        if probability >= (average_probability*0.8):
            name = main_path + '/' + labels[toGen] + '/' + labels[toGen] + '_' + str(created) + '.png'
            gen_image.save(name, 'PNG')
            created += 1

