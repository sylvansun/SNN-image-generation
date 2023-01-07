import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("-c", "--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("-s", "--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("-o", "--output_dir", type=str, default="images")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--gen", choices=["front", "mid", "back", "ann", "modular"], default="modular")
    parser.add_argument("--dis", choices=["snn", "ann"], default="ann")
    parser.add_argument(
        "-gc", "--gen_channels", type=list, default=[128, 128, 64], help="number of channels in each layer of generator"
    )
    return parser
