"""Generate ARC augmented dataset."""

import argparse
import os.path

import pyarrow
import pyarrow.parquet
import tqdm

import arc.dataset
import arc.transform


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("challanges", type=str, help="Path to challanges.")
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "--solutions", type=str, default=None, help="Path to solutions."
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=30 * 31 * 4 + 5,  # 2 example of max size w/ run 1
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-num-samples", type=int, default=1000)
    parser.add_argument(
        "--value-permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fliplr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--flipud", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--rotate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--limit-run", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-example-samples", type=int, default=-1)
    parser.add_argument("--max-run-length", type=int, default=30)
    return parser


def main():
    args = create_argparser().parse_args()
    dataset = arc.dataset.ARCDataset(
        [(args.challanges, args.solutions)],
        max_sequence_length=args.max_sequence_length,
        train=args.solutions is not None,
        tokenizer=arc.tokenizer.ARCTokenizer(
            max_run_length=args.max_run_length
        ),
        augment=arc.transform.Augment(
            value_permutation=args.value_permutation,
            fliplr=args.fliplr,
            flipud=args.flipud,
            rotate=args.rotate,
            num_example_samples=args.num_example_samples,
            limit_run=args.limit_run,
        ),
        return_raw=False,
    )
    for i in tqdm.tqdm(range(len(dataset))):
        samples = {}
        num_samples = 0
        problem = None
        while (
            len(samples) < args.num_samples
            and num_samples < args.max_num_samples
        ):
            for sample in dataset[i]:
                sequence = sample["tokenized_sequence"]
                samples[",".join(str(s) for s in sequence)] = sequence.tolist()
                problem = sample["problem"]

        table = pyarrow.table(
            [[problem] * len(samples), list(samples.values())],
            names=["problem", "tokenized_sequence"],
        )
        pyarrow.parquet.write_table(
            table, os.path.join(args.output_path, f"{problem}.parquet")
        )


if __name__ == "__main__":
    main()
