from config import *
import os
from run_dcmn_geo_4 import evaluate


def do_evaluate():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--file",
        default=None,
        type=str,
        required=True,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--model_choices",
        default=300,
        # default=128,
        type=int)

    parser.add_argument(
        "--p_num",
        default=3,
        required=True,
        # default=128,
        type=int)

    parser.add_argument(
        "--model_dir",
        default='bert',
        type=str,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--start_file",
        default='',
        type=str,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--end_file",
        default='',
        type=str,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--batch_size",
        default=11,
        type=int,
        help="fine-tuned model directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="fine-tuned model directory",
    )
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Whether to run eval on the dev set.")

    args = parser.parse_args()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_dir)
    model = model_class.from_pretrained(args.model_dir,
                                        )

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)
    args.n_gpu = 1
    args.device = device
    # model.cuda()
    model.to(device)

    if args.do_eval:
        args.dev_file = args.file
        result, output, labels = evaluate(args, model, tokenizer, mode='dev')
    elif args.do_test:
        args.test_file = args.file
        result, output, labels = evaluate(args, model, tokenizer, mode='test')





if __name__ == "__main__":
    do_evaluate()
