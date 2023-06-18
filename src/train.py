import os
import sys
import argparse

sys.path.insert(0,r'./') #Add root directory here
import torchvision.transforms as transforms
from src.data.dataloader import STDataloader
from src.models.trainer import Trainer
from src.utils.custom_transform import RGBToGrayscaleStacked, AddGaussianNoise


def parse_args(args):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--output_dir', type=str, help="The output directory to save")
    parser.add_argument('--content_datapath', nargs='+', type=str,
                        help="The path to content images dir (can be a list of paths)")
    parser.add_argument('--style_datapath', nargs='+', type=str,
                        help="The path to style images dir (can be a list of paths)")
    parser.add_argument('--eval_contentpath', type=str, default="./src/data/eval_dir/content",
                        help="The path to eval dir")
    parser.add_argument('--eval_stylepath', type=str, default="./src/data/eval_dir/style",
                        help="The path to style dir")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for the dataloader")
    parser.add_argument('--eval_batch_size', type=int, default=1, help="Eval batch size for the dataloader")
    parser.add_argument('--max_content_train_samples', type=int, default=None, help="Number of content training samples")
    parser.add_argument('--max_style_train_samples', type=int, default=None, help="Number of style training samples")
    parser.add_argument('--num_worker', type=int, default=2, help="Number of worker for dataloader")
    parser.add_argument('--seed', type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument('--crop_width', type=int, default=256, help="Width of the image randomly center crop")
    parser.add_argument('--crop_height', type=int, default=256, help="Width of the image randomly center crop")


    # Training
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help="number training epochs")
    parser.add_argument('--step_frequency', type=float, default=0.5,
                        help="How often should you update the lr (Should be a fraction of an epoch)")
    parser.add_argument('--with_tracking', action='store_true',
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument('--log_weights_cpkt', action='store_true',
                        help="Whether to log weights to tracker")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help="If the training should continue from a checkpoint folder. (can be bool or string)")
    parser.add_argument('--login_key', type=str, default=None,
                        help="Login key for wandb")
    parser.add_argument('--plot_per_epoch', action='store_true',
                        help="Whether to plot comparison per epoch.")
    parser.add_argument('--do_eval_per_epoch', action='store_true',
                        help="Whether to run evaluate per epoch.")
    parser.add_argument('--transformer_size', type=int, default=32,
                        help="Size of the transformation matrix")
    parser.add_argument('--CNN_layer_depth', type=int, default=1,
                        help="Depth of CNN layer")
    parser.add_argument('--deep_learner', action='store_true',
                        help="Whether to enable deepest layer for abstract learning.")

    # Optimizer
    parser.add_argument('--vgg_model_type', type=str, default='19',help=(
            "Which models of the vgg to use as a feature extractor"
        ))
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--warm_up_epoch', type=int, default=1,
                        help="Warmup period")
    parser.add_argument('--alpha', type=float, default=5e-5,
                        help="Initial alpha to use.")
    parser.add_argument('--beta', type=float, default=5e-5,
                        help="Initial beta to use.")
    parser.add_argument('--gamma', type=float, default=5e-5,
                        help="Initial gamma to use.")
    parser.add_argument('--delta', type=float, default=5e-5,
                        help="Initial delta to use.")
    parser.add_argument('--weight_decay', type=float, default=0.3,
                        help="Weight decay to use.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--content_layers_idx', nargs='+', type=int, default=[8, 11, 13],
                        help="Vgg layers to compute content loss")
    parser.add_argument('--style_layers_idx', nargs='+', type=int, default=[1, 3, 6, 8],
                        help="Vgg layers to compute style loss")

    args = parser.parse_args(args)

    # # validate and convert the input argument
    # try:
    #     args.checkpointing_steps = int(args.checkpointing_steps)  # try to convert to int
    # except:
    #     args.checkpointing_steps = args.checkpointing_steps  # if conversion fails, assume it's a string

    # Sanity check
    assert os.path.isdir(args.output_dir), "Invalid output dir path!"
    for path in args.content_datapath:
        assert os.path.isdir(path), "Invalid content dir path!"
    for path in args.style_datapath:
        assert os.path.isdir(path), "Invalid style dir path!"
    assert os.path.isfile(args.resume_from_checkpoint) \
        if args.resume_from_checkpoint is not None else True , "Invalid cpkt path!"

    return args


def main(args):
    args = parse_args(args)
    dataloader_args = {
        "content_datapath": args.content_datapath,
        "style_datapath": args.style_datapath,
        "eval_contentpath": args.eval_contentpath,
        "eval_stylepath": args.eval_stylepath,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "transform_content": transforms.Compose([
            transforms.CenterCrop((args.crop_width, args.crop_height)),
            transforms.ToTensor(),
            RGBToGrayscaleStacked(),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5)
        ]),
        "transform_style": transforms.Compose([
            transforms.CenterCrop((args.crop_width, args.crop_height)),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.5, sigma_range=(0., 0.08), p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5)
        ]),
        "max_style_train_samples": args.max_style_train_samples,
        "max_content_train_samples": args.max_content_train_samples,
        "seed": args.seed,
        "num_worker": args.num_worker
    }
    dataloaders = STDataloader(**dataloader_args)

    trainer_args = {
        "dataloaders": dataloaders,
        "output_dir": args.output_dir,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "with_tracking": args.with_tracking,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "per_device_batch_size":dataloaders.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "do_eval_per_epoch": args.do_eval_per_epoch,
        "plot_per_epoch": args.plot_per_epoch,
        "warm_up_epoch": args.warm_up_epoch,
        "learning_rate": args.learning_rate,
        "vgg_model_type": args.vgg_model_type,
        "transformer_size": args.transformer_size,
        "layer_depth": args.CNN_layer_depth,
        "deep_learner": args.deep_learner,
        "login_key": args.login_key,
        "seed": args.seed,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "content_layers_idx": args.content_layers_idx,
        "style_layers_idx": args.style_layers_idx,
        "log_weights_cpkt": args.log_weights_cpkt,
        "step_frequency": args.step_frequency
    }
    trainer = Trainer(**trainer_args)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
