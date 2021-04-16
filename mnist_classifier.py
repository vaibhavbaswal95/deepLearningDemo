
import numpy as np
from argparse import ArgumentParser
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
        #initial parameters
        self.img_shape = img_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # defining a helper conv block with conv2d and batch normalize layer
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
        #forward pass
        step1 = self.block1(x)
        log(f"block1 has size {step1.size()}")
        step2 = self.block2(step1)
        log(f"block1 has size {step2.size()}")
        x = self.block3(step2)
        log(f"block1 has size {x.size()}") 
        return x

def testNet():
    #testing the defined architechture
    data_shape = (2, 1, 28, 28)
    test_net = MnistNet(data_shape)
    #random sample
    sample = torch.rand(data_shape)
    test_net.eval()
    a = test_net.forward(sample)
    log(a[0].size) #logging
# testNet()

## Model Class inhereting PyTorch LightningModule
class MnistClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(MnistClassifier, self).__init__()
        #cache variables
        self.hparams = hparams
        self.img_shape = (28,28,1)
        self.batch_size = 64
        self.transform = None
        self.root = './'
        self.mnistNet = MnistNet(self.img_shape)
        self.loss = torch.nn.CrossEntropyLoss()
        self.test_accuracy = []

    def train_dataloader(self):
        # composing all the transfroms 
        # data augmentation happens here 
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=0.5,std=0.5)])
        
        self.dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform,)
        #returning dataloader
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        # composing all the transfroms 
        # data augmentation happens here 
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=0.5,std=0.5)])
        
        self.dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform,)
        #returning dataloader
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def forward(self, x):
        #forward pass
        return self.mnistNet(x)

    def training_step(self, batch, batch_idx):
        # training step, called everytime with a batch
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y) 

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y) 
        # getting the class from probabilities
        label_hat = torch.argmax(y_hat, dim=1)
        
        test_acc = torch.sum(y==label_hat).item()/(len(y)*1.0)
        log(f"Test Accuracy : {test_acc}")
        return label_hat

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # run on test batch end
        x,y = batch
        y = y.to('cuda')
        # caching the batchwise accuracy
        self.test_accuracy.append(torch.sum(y==outputs).item()/(len(y)*1.0))
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_test_end(self):
        # run on test end
        # Mean of Batchwise Accuracy
        log(f"Mean Test Accuracy Batchwise : {np.asarray(self.test_accuracy).mean()}")
        return super().on_test_end()

    def configure_optimizers(self):
        # Optimizers to be defined here
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main(hparams):
    log("Into Main Function")
    print("######################################TRAIN######################################")
    # Loading the model
    mnist_classifier = MnistClassifier(hparams)
    train_dataloader = mnist_classifier.train_dataloader()
    test_dataloader = mnist_classifier.test_dataloader()
    # defiining the trainer
    trainer = pl.Trainer(gpus=[0], max_epochs=hparams.epochs, resume_from_checkpoint='/home/vaibhav/Desktop/deloitte/mnist_checkpoint.ckpt')
    #training
    trainer.fit(mnist_classifier, train_dataloader)
    #saving the model
    trainer.save_checkpoint("./mnist_checkpoint.ckpt")

    print("######################################TEST######################################")
    #loading the model for testing
    mnist_for_inference = MnistClassifier.load_from_checkpoint(checkpoint_path="./mnist_checkpoint.ckpt")
    #changing the mode to eval > for Dropouts Layers, BatchNorm Layers
    mnist_for_inference.eval()
    trainer = pl.Trainer(gpus=[0],)
    trainer.test(model = mnist_for_inference, test_dataloaders=test_dataloader,)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="size of the batches")
    parser.add_argument("--epochs",
                    type=int,
                    default=10,
                    help="Number of Epochs to Train")
    #more arguments can be added here as hparams
    # these hparams are also logged by default in the lightning_log directory
    hparams = parser.parse_args()
    main(hparams)

