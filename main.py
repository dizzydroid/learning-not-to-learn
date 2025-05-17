# main.py

import argparse
import torch
import os

from data_handler import get_data_loader
from network import FeatExtractor, LabelClassifier, BiasClassifier
from solver import Solver
from utils import seed_everything

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    seed_everything(args.seed)

    print("Preparing data loaders...")
    train_loader = get_data_loader(args, train=True, biased_training_target_for_mnist=args.cmnist_train_bias_type)
    val_loader = None
    if args.do_eval:
        val_loader = get_data_loader(args, train=False, biased_training_target_for_mnist=False)

    print("Initializing networks...")
    model_f = FeatExtractor(network_name=args.f_network_name,
                            input_channels=args.input_channels,
                            pretrained=args.f_pretrained).to(device)
    model_g = LabelClassifier(network_name=args.g_network_name,
                              num_classes=args.num_main_classes,
                              input_channels_f=model_f.output_channels,
                              input_dim_spatial_f=getattr(model_f, 'output_dim_spatial', None)
                              ).to(device)
    
    if args.h_network_name is None:
        if args.dataset_name == 'ColoredMNIST': h_network_name = 'SimpleCNN_ConvBias'
        elif args.dataset_name == 'IMDBFace': h_network_name = 'ResNet18_FCBias'
        elif args.dataset_name == 'DogsAndCats': h_network_name = 'ResNet18_FCBias'
        else: raise ValueError("Cannot infer h_network_name")
    else: h_network_name = args.h_network_name

    model_h = BiasClassifier(network_name=h_network_name,
                             num_bias_classes=args.num_bias_classes,
                             input_channels_f=model_f.output_channels,
                             input_dim_spatial_f=getattr(model_f, 'output_dim_spatial', None)
                             ).to(device)

    print("Initializing solver...")
    solver = Solver(args, train_loader, val_loader, model_f, model_g, model_h, device)

    if args.mode == 'train':
        if args.load_checkpoint_path: solver.load_model(args.load_checkpoint_path)
        solver.train()
    elif args.mode == 'eval' or args.mode == 'test': # Added 'test' mode
        if not args.load_checkpoint_path:
            print("Error: Checkpoint path must be provided for eval/test mode.")
            return
        start_epoch = solver.load_model(args.load_checkpoint_path) # Loads model weights
        if args.mode == 'eval':
            solver.evaluate(epoch_num=start_epoch if start_epoch is not None else 0)
        elif args.mode == 'test':
            # Assuming solver has a method like 'test_and_save_results'
            if hasattr(solver, 'test_and_save_results'):
                print("Running test mode and saving results...")
                solver.test_and_save_results(output_npy_file=args.test_output_npy)
            else:
                print("Solver does not have 'test_and_save_results' method. Performing standard evaluation.")
                solver.evaluate(epoch_num=start_epoch if start_epoch is not None else 0)

    else:
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'train', 'eval', or 'test'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learning Not to Learn - Main Runner")

    # General
    parser.add_argument('--experiment_name', type=str, default="default_exp")
    parser.add_argument('--output_dir', type=str, default="./outputs")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test']) # Added 'test'
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    parser.add_argument('--test_output_npy', type=str, default="test_results.npy", help="Filename for saving test results (if mode=test)")


    # Dataset
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ColoredMNIST', 'DogsAndCats', 'IMDBFace'])
    # Changed data_root to data_root_base to reflect it's the parent './data' directory
    parser.add_argument('--data_root_base', type=str, default="./data", help="Base directory for all datasets (e.g., ./data)")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--download_data', type=bool, default=True, help="Download dataset if applicable (MNIST)")

    # CMNIST specific
    parser.add_argument('--sigma_bias', type=float, default=0.05)
    parser.add_argument('--cmnist_bias_label_type', type=str, default='mean_color_idx')
    parser.add_argument('--cmnist_train_bias_type', type=bool, default=True)

    # D&C specific (using list files)
    parser.add_argument('--dac_list_bright', type=str, default="list_bright.txt", help="Filename for bright images list (in data_root_base/dogs_vs_cats/lists/)")
    parser.add_argument('--dac_list_dark', type=str, default="list_dark.txt", help="Filename for dark images list")
    parser.add_argument('--dac_list_test', type=str, default="list_test_unbiased.txt", help="Filename for unbiased test images list")
    
    # IMDB / General D&C manifest (still useful for IMDB or if D&C uses manifest)
    parser.add_argument('--train_manifest', type=str, default=None, help="Filename for train manifest (e.g. for IMDB, in data_root_base/imdb_face/manifests/)")
    parser.add_argument('--test_manifest', type=str, default=None, help="Filename for test/val manifest")
    parser.add_argument('--bias_type_train', type=str, default=None, help="e.g., TB1, TB2 for D&C; EB1, EB2 for IMDB")
    parser.add_argument('--imdb_task', type=str, default='classify_gender', choices=['classify_gender', 'classify_age'])

    # Network
    parser.add_argument('--f_network_name', type=str, default='ResNet18', choices=['SimpleCNN', 'ResNet18'])
    parser.add_argument('--f_pretrained', type=bool, default=True)
    parser.add_argument('--g_network_name', type=str, default='ResNet18', choices=['SimpleCNN', 'ResNet18'])
    parser.add_argument('--h_network_name', type=str, default=None)
    parser.add_argument('--num_main_classes', type=int, required=True)
    parser.add_argument('--num_bias_classes', type=int, required=True)

    # Training (Solver)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrain_fg_epochs', type=int, default=0)
    parser.add_argument('--lr_fg', type=float, default=1e-4)
    parser.add_argument('--lr_h', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--adv_mu', type=float, default=1.0)
    parser.add_argument('--adv_lambda', type=float, default=0.5)
    parser.add_argument('--grl_gamma', type=float, default=10.0)
    parser.add_argument('--save_every_epochs', type=int, default=10)
    parser.add_argument('--do_eval', type=bool, default=True)

    args = parser.parse_args()

    # Argument post-processing/validation
    if args.dataset_name == 'ColoredMNIST':
        args.f_network_name = 'SimpleCNN'; args.g_network_name = 'SimpleCNN'
        args.input_channels = 3; args.image_size = 28
    elif args.dataset_name == 'DogsAndCats' or args.dataset_name == 'IMDBFace':
        if args.f_network_name == 'SimpleCNN' or args.g_network_name == 'SimpleCNN':
            print(f"Warning: Using SimpleCNN for {args.dataset_name}. ResNet18 is typical.")
    
    # Ensure manifest paths are constructed correctly for IMDB if specified
    # (get_data_loader now handles joining with data_root_base/dataset_name/manifests/)

    main(args)
