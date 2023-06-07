import os
import sys
import argparse

sys.path.insert(0,r'./') #Add root directory here
import torchvision.transforms as transforms
from src.data.dataloader import STDataloader
from src.models.trainer import Trainer


def parse_args(args):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--output_dir', type=str, help="The output directory to save")
    parser.add_argument('--content_datapath', type=str, help="The path to content images dir")
    parser.add_argument('--style_datapath', type=str, help="The path to style images dir")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for the dataloader")
    parser.add_argument('--max_content_train_samples', type=int, default=None, help="Number of content training samples")
    parser.add_argument('--max_style_train_samples', type=int, default=None, help="Number of style training samples")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="Number of validation samples")
    parser.add_argument('--seed', type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument('--width', type=int, default=128, help="Width of the image")
    parser.add_argument('--height', type=int, default=128, help="Height of the image")

    # Training
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help="number training epochs")
    parser.add_argument('--with_tracking', action='store_true',
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help="If the training should continue from a checkpoint folder. (can be bool or string)")
    parser.add_argument('--do_eval_per_epoch', action='store_true',
                        help="Whether to run evaluate per epoch.")

    # Optimizer
    parser.add_argument('--vgg_model_type', type=str, default='19',help=(
            "Which models of the vgg to use as a feature extractor"
        ))
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--alpha', type=float, default=5e-5,
                        help="Initial alpha to use.")
    parser.add_argument('--beta', type=float, default=5e-5,
                        help="Initial beta to use.")
    parser.add_argument('--gamma', type=float, default=5e-5,
                        help="Initial gamma to use.")
    parser.add_argument('--weight_decay', type=float, default=0.3,
                        help="Weight decay to use.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)

    args = parser.parse_args(args)

    # # validate and convert the input argument
    # try:
    #     args.checkpointing_steps = int(args.checkpointing_steps)  # try to convert to int
    # except:
    #     args.checkpointing_steps = args.checkpointing_steps  # if conversion fails, assume it's a string

    return args


def main(args):

    args = parse_args(args)
    dataloader_args = {
        "content_datapath": args.content_datapath,
        "style_datapath": args.style_datapath,
        "batch_size": args.batch_size,
        "transform": transforms.Compose([
            transforms.Resize((args.width, args.height)),
            transforms.ToTensor(),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.37, hue=0.42),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.CenterCrop()
        ]),
        "max_style_train_samples": args.max_style_train_samples,
        "max_content_train_samples": args.max_content_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "seed": args.seed
    }
    dataloaders = STDataloader(**dataloader_args)

    trainer_args = {
        "dataloaders": dataloaders,
        "output_dir": args.output_dir,
        "lr_scheduler_type": args.lr_scheduler_type,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "with_tracking": args.with_tracking,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "per_device_batch_size":dataloaders.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "do_eval_per_epoch": args.do_eval_per_epoch,
        "learning_rate": args.learning_rate,
        "vgg_model_type": args.vgg_model_type,
        "login_key": None,
        "seed": args.seed,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma
    }
    trainer = Trainer(**trainer_args)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
