import argparse


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description="Split Learning Research Simulation for medical datasets entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--number_of_clients",
        type=int,
        default=6,
        metavar="C",
        help="Number of Clients",
    )
    
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=2,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "-tbs",
        "--test_batch_size",
        type=int,
        default=2,
        metavar="TB",
        help="Input batch size for testing",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Total number of epochs to train",
    )

    parser.add_argument(
        "--client_lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Client-side learning rate",
    )
    
    parser.add_argument(
        "--server_lr",
        type=float,
        default=1e-3,
        metavar="serverLR",
        help="Server-side learning rate"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="kits19",
        help="States dataset to be used",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="nnunet",
        help="Model you would like to train",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="The model split version to use"
    )

    parser.add_argument(
        '-kv',
        '--use_key_value_store',
        action="store_true",
        default=False,
        help="use key value store for faster training"
    )

    parser.add_argument(
        '--kv_factor',
        type=int,
        default=1,
        help="populate key value store kv_factor times"
    )

    parser.add_argument(
        '--kv_refresh_rate',
        type=int,
        default=5,
        help="refresh key-value store every kv_refresh_rate epochs, 0 = disable refresing"
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable wandb logging",
    )
    
    parser.add_argument(
        "--wandb_name",
        help="WANDB PROJECT NAME"
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Model is pretrained/not, DEFAULT True, No change required",
    )

    parser.add_argument(
        "--personalize",
        action="store_true",
        default=False,
        help="Enable client personalization"
    )

    parser.add_argument(
        '--pool',
        action="store_true",
        default=False,
        help="create a single client with all the data, trained in split learning mode, overrides number_of_clients"
    )

    parser.add_argument(
        '--dynamic',
        action="store_true",
        default=False,
        help="Use dynamic transforms, transforms will be applied to the server-side kv-store every epoch"
    )

    parser.add_argument(
        "--p_epoch",
        type=int,
        default=50,
        help="Epoch at which personalisation phase will start",
    )

    parser.add_argument(
        "--offload_only",
        action="store_true",
        default=False,
        help="USE SERVER ONLY FOR OFFLOADING, CURRENTLY ONLY IMPLEMENTED FOR IXI-TINY & KITS19",
    )


    args = parser.parse_args()
    return args
