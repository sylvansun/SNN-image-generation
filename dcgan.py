import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

from utils.parser import make_parser
from utils.etqdm import etqdm
from utils.dataset import get_dataset
from models.model_zoo import Gen, GenFront, GenMid, GenBack, Dis, DisSpike


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def make_generator(args):
    models = {"front": GenFront(args), "mid": GenMid(args), "back": GenBack(args), "ann": Gen(args)}
    return models[args.gen]

def make_discriminator(args):
    models = {"ann": Dis(args), "snn": DisSpike(args)}
    return models[args.dis]


def main(args):
    adversarial_loss = torch.nn.BCELoss()
    generator = make_generator(args)
    discriminator = make_discriminator(args)
    save_img_dir = os.path.join(args.output_dir, "dcgan" , args.gen + "_" + args.dis)
    os.makedirs(save_img_dir, exist_ok=True)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    dataloader,_ = get_dataset(args.batch_size, args.img_size)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))

            for _ in range(1):
                optimizer_G.zero_grad()
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        save_image(gen_imgs.data[:25], os.path.join(save_img_dir , f"{epoch}.png"), nrow=5, normalize=True)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    cuda = True if torch.cuda.is_available() else False
    
    main(args)