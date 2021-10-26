import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="facebook/bart-base",
        choices=[
            "facebook/bart-base",
            "facebook/bart-large",
            "t5-small",
            "t5-base",
            "t5-large",
            "p208p2002/bart-squad-qg-hl",
            "p208p2002/bart-squad-nqg-hl",
            "p208p2002/t5-squad-qg-hl",
            "p208p2002/t5-squad-nqg-hl",
        ],
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="seq2seq_lm",
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--data_type", default="squad", choices=["squad", "race"], type=str
    )
    parser.add_argument(
        "--train_file", default="None", type=str, help="The input training file."
    )
    parser.add_argument(
        "--dev_file", default="None", type=str, help="The input dev file."
    )
    parser.add_argument(
        "--predict_file", default="None", type=str, help="The input evaluation file."
    )
    parser.add_argument("--task_name", default="task", choices=["seq2seq_QG"], type=str)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size for training/testing.",
    )
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("-fc", "--from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    return args