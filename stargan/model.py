"""
model.py

Author - Max Elliott

Contains the full proposed model (the adapted StarGAN-VC model for spectral
envelope conversion).

Note: Generator_World is the generator used for all experiments and the one
described in the report. Generator_Mel was only an alternative that never got
used.
"""
import os
from stargan.classifiers import *

import stargan.unet.unet_model as unet_model


class StarGAN_emo_VC1(object):
    """
    The proposed model of this project.
    """
    def __init__(self, config, name):
        """
        Need config for input_size, hidden_size, num_layers, num_classes, bi = False
        """
        super(StarGAN_emo_VC1, self).__init__()
        self.config = config
        self.save_dir = config['logs']['model_save_dir']
        self.name = name
        self.use_speaker = config['model']['use_speaker']
        self.use_dimension = config['model']['use_dimension']

        self.build_model()

    def set_train_mode(self):
        self.G.train()
        self.D.train()
        self.emo_cls.train()

        if self.use_speaker:
            self.speaker_cls.train()
        if self.use_dimension:
            self.dimension_cls.train()

    def set_eval_mode(self):
        self.G.eval()
        self.D.eval()
        self.emo_cls.eval()

        if self.use_speaker:
            self.speaker_cls.eval()
        if self.use_dimension:
            self.dimension_cls.eval()

    def to_device(self, device=torch.device('cuda')):
        if torch.cuda.is_available():
            self.G.to(device=device)
            self.D.to(device=device)
            self.emo_cls.to(device=device)
            self.emo_cls.device = device

            if self.use_speaker:
                self.speaker_cls.to(device=device)
            if self.use_dimension:
                self.dimension_cls.to(device=device)
        else:
            print("Device not available")
            self.G.to(device=torch.device('cpu'))
            self.D.to(device=torch.device('cpu'))
            self.emo_cls.to(device=torch.device('cpu'))
            self.emo_cls.device = device

            if self.use_speaker:
                self.speaker_cls.to(device=torch.device('cpu'))
            if self.use_dimension:
                self.dimension_cls.to(device=torch.device('cpu'))

    def build_model(self):

        self.num_input_feats = self.config['model']['num_feats']
        self.hidden_size = 128
        self.num_layers = 2
        self.num_emotions = self.config['model']['num_classes']
        self.num_speakers = 10
        self.bi = True

        print("Building components")

        if self.config['data']['type'] == 'mel':
            self.G = nn.DataParallel(Generator_Mel(self.num_emotions))
        else:
            self.G = nn.DataParallel(Generator_World(self.num_emotions))
        self.D = nn.DataParallel(Discriminator(self.num_emotions))
        self.emo_cls = nn.DataParallel(Emotion_Classifier(self.num_input_feats,
                                                          self.hidden_size,
                                                          self.num_layers,
                                                          self.num_emotions,
                                                          bi=self.bi))

        self.speaker_cls = nn.DataParallel(Emotion_Classifier(self.num_input_feats,
                                                              self.hidden_size,
                                                              self.num_layers,
                                                              self.num_speakers,
                                                              bi=self.bi))

        self.dimension_cls = nn.DataParallel(Dimension_Classifier(self.num_input_feats,
                                                                  self.hidden_size,
                                                                  self.num_layers,
                                                                  bi=self.bi))

        print("Building optimizers")

        con_opt = self.config['optimizer']
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), con_opt['g_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), con_opt['d_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.emo_cls_optimizer = torch.optim.Adam(self.emo_cls.parameters(), con_opt['emo_cls_lr'], [con_opt['beta1'], con_opt['beta2']],weight_decay=0.000001)
        if self.use_speaker:
            self.speaker_cls_optimizer = torch.optim.Adam(self.speaker_cls.parameters(), con_opt['speaker_cls_lr'], [con_opt['beta1'], con_opt['beta2']])
        if self.use_dimension:
            self.dimension_cls_optimizer = torch.optim.Adam(self.dimension_cls.parameters(), con_opt['dim_cls_lr'], [con_opt['beta1'], con_opt['beta2']])

        if self.config['verbose']:
            print("Network parameter list:")
            total = 0
            G_count = self.print_network(self.G, 'G')
            D_count = self.print_network(self.D, 'D')
            emo_count = self.print_network(self.emo_cls, 'C_emotion')
            if self.use_speaker:
                spk_count = self.print_network(self.speaker_cls, 'C_Speaker')
                total += spk_count
            if self.use_dimension:
                dim_count = self.print_network(self.dim_cls, 'C_Dimensional')
                total += dim_count

            total += G_count + D_count + emo_count

            print("TOTAL NUMBER OF PARAMETERS = {}".format(total))

        self.to_device(device='cpu')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        return num_params

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.emo_cls_optimizer.zero_grad()
        if self.use_speaker:
            self.speaker_cls_optimizer.zero_grad()
        if self.use_dimension:
            self.dimension_cls_optimizer.zero_grad()

    def save(self, save_dir=None, it=0):

        if save_dir is None:
            save_dir = self.save_dir

        path = os.path.join(save_dir, self.name)
        if not os.path.exists(path):
            os.makedirs(path)

        self.config['loss']['resume_iters'] = it

        state = {'D': self.D.state_dict(),
                 'G': self.G.state_dict(),
                 'emo': self.emo_cls.state_dict(),
                 'd_opt': self.d_optimizer.state_dict(),
                 'g_opt': self.g_optimizer.state_dict(),
                 'emo_opt': self.emo_cls_optimizer.state_dict(),
                 'config': self.config
                 }

        if self.use_speaker:
            state['spk'] = self.speaker_cls.state_dict()
            state['spk_opt'] = self.speaker_cls_optimizer.state_dict()
        if self.use_dimension:
            state['dim'] = self.dimension_cls.state_dict()
            state['dim_opt'] = self.dimension_cls_cls_optimizer.state_dict()

        path = os.path.join(path, "{:06}.ckpt".format(it))

        torch.save(state, path)

        print("Model saved as {}.".format(path))

    def load(self, load_dir):
        """
        load_dir: full directory of checkpoint to load
        """
        # if load_dir[-1] == '/':
        #     load_dir = load_dir[0:-1]
        #
        # self.name = os.path.basename(load_dir)
        #
        # path = os.path.join(load_dir, "{:05}.ckpt".format(iter))

        print(load_dir)

        # if map_location is not None:
        #     dictionary = torch.load(load_dir, map_location=map_location)
        # else:
        #     dictionary = torch.load(load_dir)

        if torch.cuda.is_available():
            dictionary = torch.load(load_dir)
        else:
            dictionary = torch.load(load_dir, map_location='cpu')

        self.config = dictionary['config']
        self.use_speaker = self.config['model']['use_speaker']
        self.use_dimension = self.config['model']['use_dimension']
        self.name = self.config['model']['name']
        self.num_emotions = self.config['model']['num_classes']

        self.D.load_state_dict(dictionary['D'])
        self.G.load_state_dict(dictionary['G'])
        self.emo_cls.load_state_dict(dictionary['emo'])

        # self.d_optimizer.load_state_dict(dictionary['d_opt'])
        # self.g_optimizer.load_state_dict(dictionary['g_opt'])
        # self.emo_cls_optimizer.load_state_dict(dictionary['emo_opt'])

        con_opt = self.config['optimizer']
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), con_opt['g_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), con_opt['d_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.emo_cls_optimizer = torch.optim.Adam(self.emo_cls.parameters(), con_opt['emo_cls_lr'], [con_opt['beta1'], con_opt['beta2']], weight_decay=0.000001)

        if 'spk' in dictionary:
            self.speaker_cls.load_state_dict(dictionary['spk'])
            self.speaker_cls_optimizer.load_state_dict(dictionary['spk_opt'])
            self.use_speaker = True
        else:
            self.use_speaker = False
        if 'dim' in dictionary:
            self.dimension_cls.load_state_dict(dictionary['dim'])
            self.dimension_cls_optimizer.load_state_dict(dictionary['dim_opt'])
            self.use_dimension = True
        else:
            self.use_dimension = False

        print("Model and optimizers loaded.")

    def load_pretrained_classifier(self, load_dir, map_location=None):

        if map_location is not None:
            dictionary = torch.load(load_dir, map_location=map_location)
        else:
            dictionary = torch.load(load_dir)

        con_opt = self.config['optimizer']
        self.emo_cls.load_state_dict(dictionary['model_state_dict'])
        self.emo_cls_optimizer = torch.optim.Adam(self.emo_cls.parameters(), con_opt['emo_cls_lr'], [con_opt['beta1'], con_opt['beta2']],weight_decay = 0.000001)


class Down2d(nn.Module):
    """docstring for Down2d."""
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 = x1 * torch.sigmoid(x2)

        return x3


class Up2d(nn.Module):
    """docstring for Up2d."""
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 = x1 * torch.sigmoid(x2)

        return x3


class Generator_World(nn.Module):
    """docstring for Generator."""
    def __init__(self, num_classes=4):
        super(Generator_World, self).__init__()

        self.num_classes = num_classes

        self.down1 = Down2d(1, 32, (9, 3), (1, 1), (4, 1))
        self.down2 = Down2d(32, 64, (8, 4), (2, 2), (3, 1))
        self.down3 = Down2d(64, 128, (8, 4), (2, 2), (3, 1))
        self.down4 = Down2d(128, 64, (5, 3), (1, 1), (2, 1))
        self.down5 = Down2d(64, 5, (5, 9), (1, 9), (2, 0))

        self.up1 = Up2d(5 + num_classes, 64, (5, 9), (1, 9), (2, 0))
        self.up2 = Up2d(64 + num_classes, 128, (5, 3), (1, 1), (2, 1))
        self.up3 = Up2d(128 + num_classes, 64, (8, 4), (2, 2), (3, 1))
        self.up4 = Up2d(64 + num_classes, 32, (8, 4), (2, 2), (3, 1))

        self.deconv = nn.ConvTranspose2d(32 + num_classes, 1, (9, 3), (1, 1), (4, 1))

    def forward(self, x, c):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))

        x = torch.cat([x, c1], dim=1)

        x = self.up1(x)
        c2 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)
        c3 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)
        c4 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.up4(x)
        c5 = c.repeat(1,1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.deconv(x)

        return x


class Generator_Mel(nn.Module):
    def __init__(self, num_classes):
        super(Generator_Mel, self).__init__()
        self.unet = unet_model.UNet(1,1)

    def forward(self, x, c):
        x = self.unet(x)
        return x


class Discriminator(nn.Module):
    """docstring for Discriminator."""
    def __init__(self, num_classes=4):
        super(Discriminator, self).__init__()

        self.d1 = Down2d(1 + num_classes, 32, (9, 3), (1, 1), (4, 1))
        self.d2 = Down2d(32 + num_classes, 32, (8, 3), (2, 1), (3, 1))
        self.d3 = Down2d(32 + num_classes, 32, (8, 3), (2, 1), (3, 1))
        self.d4 = Down2d(32 + num_classes, 32, (6, 3), (2, 1), (2, 1))

        self.conv = nn.Conv2d(32 + num_classes, 1, (5, 36), (1, 36), (2, 0))
        self.pool = nn.AvgPool2d((64,1))

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)
        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.d3(x)
        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.d4(x)
        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.tanh(x)

        return x


if __name__ == '__main__':
    pass
    # import yaml
    # config = yaml.load(open('./config.yaml', 'r'))
    #
    # l = [0,0,0,1,2,3,4,5]
    #
    # l = [v for v in l if v in [0,1,5]]
    #
    # print(l)
