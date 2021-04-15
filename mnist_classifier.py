
import idx2numpy
import numpy as np
from argparse import ArgumentParser
import torch
import torchvision
import pytorch_lightning as pl

import logging
# create logger
logger = logging.getLogger(__name__)
#setting logging level
level = logging.INFO
logger.setLevel(level)
#adding output stream
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# add ch to logger
logger.addHandler(ch)

def log(log_str):
    logger.info(f"INFO > {log_str}")
    return None

class MnistNet(torch.nn.Module):
    def __init__(self, img_shape, kernel_size=3, stride=1, padding=1):
        super(MnistNet, self).__init__()
        self.img_shape = img_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        def conv_block(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding ,normalize=True):
                layers = [torch.nn.Conv2d(in_feat, out_feat, kernel_size, padding = padding, stride=stride, bias=False)]
                
                if normalize:
                    layers.append(torch.nn.BatchNorm2d(out_feat))
                layers.append(torch.nn.LeakyReLU(inplace=True))

                return layers
        
        self.base_filters_num = 2

        self.block1 = torch.nn.Sequential(
                *conv_block(1, self.base_filters_num, stride=2),
                *conv_block(self.base_filters_num, self.base_filters_num*2, stride=2),
            )

        self.block2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            )

        self.block3 = torch.nn.Sequential(
            torch.nn.Linear(196, 10),
            )

    def forward(self, x):
        step1 = self.block1(x)
        log(f"block1 has size {step1.size()}")
        step2 = self.block2(step1)
        log(f"block1 has size {step2.size()}")
        x = self.block3(step2)
        log(f"block1 has size {x.size()}") 
        return x

def testNet():
    data_shape = (2, 1, 28, 28)
    test_net = MnistNet(data_shape)
    sample = torch.rand(data_shape)
    test_net.eval()
    a = test_net.forward(sample)
    logger.info(a[0].size)
# testNet()


class MNIST_CLASSIFIER(pl.LightningModule):
    def __init__(self, hparams):
        super(MNIST_CLASSIFIER, self).__init__()
        self.hparams = hparams
        self.img_shape = (28,28,1)
        self.batch_size = None
        self.transform = None
        self.root = './'
        self.mnistNet = MnistNet(self.img_shape)
        #downloading dataset
        self.dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        #creating a dataloader with custom batch_size
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def forward(self):
        for batch in iter(self.dataloader):
            log(f"Image size is > {batch[0].size}")
            break
        #  return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self(x)
         loss = F.cross_entropy(y_hat, y)
         return loss

    def configure_optimizers(self):
         return torch.optim.Adam(self.parameters(), lr=0.02)


def main(hparams):
    log("Into Main Function")
    mnist_classifier = MNIST_CLASSIFIER(hparams)
    mnist_classifier.forward()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="size of the batches")

    hparams = parser.parse_args()
    main(hparams)

