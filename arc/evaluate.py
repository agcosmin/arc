import argparse

import torch
import tqdm

import sklearn.metrics

import arc.transformer
import arc.tokenizer
import arc.dataset
import arc.submission


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("challanges", type=str, help="Path to challanges.")
    parser.add_argument(
        "solutions", type=str, default=None, help="Path to solutions."
    )
    parser.add_argument("model", type=str)
    parser.add_argument("--num-samples", type=int, default=100)
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
        "--limit-run", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-example-samples", type=int, default=10)
    return parser


def main():
    args = create_argparser().parse_args()
    dataset = arc.dataset.ARCDataset([(args.challanges, args.solutions)])
    checkpoint = torch.load(args.model)
    encoder_config = checkpoint["hyper_parameters"]["config"]
    tokenizer = arc.tokenizer.ARCTokenizer(
        max_run_length=encoder_config.tokenizer_max_run_length
    )
    model = arc.transformer.ARCEncoder(encoder_config)
    model.eval()
    model.load_state_dict(
        {
            k.replace("encoder.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k != "lm_loss.weight"
        }
    )
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    augment = arc.transform.Augment(
        value_permutation=args.value_permutation,
        fliplr=args.fliplr,
        flipud=args.flipud,
        rotate=args.rotate,
        num_example_samples=args.num_example_samples,
        limit_run=args.limit_run,
    )

    labels_true = []
    labels_predicted = []
    num_wrong_shape_predictions = 0
    for problem_id, problem in tqdm.tqdm(dataset, "Solve"):
        for test in problem["test"]:
            sequence, *_ = arc.transform.sample_sequence(
                examples=problem["train"],
                augment=augment,
                tokenizer=tokenizer,
                max_length=30 * 31 * 3 + 7,
                test=test,
                generate_test=False,
                return_img_sequence=False,
            )
            solution = arc.submission.solve_problem(
                examples=problem["train"],
                test=test,
                tokenizer=tokenizer,
                model=model,
                num_samples=args.num_samples,
                device=device,
            )
            if test["output"].shape == solution.shape:
                labels_true.append(test["output"].flatten())
                labels_predicted.append(solution.flatten())
            else:
                num_wrong_shape_predictions += 1

    metrics_average_methods = ["macro", "micro", "weighted"]
    labels = list(range(10))

    # Compute metrics averaged over puzzles
    num_solved = 0
    precisions = {m: [] for m in metrics_average_methods}
    recalls = {m: [] for m in metrics_average_methods}
    for true, pred in zip(labels_true, labels_predicted):
        for method in metrics_average_methods:
            precisions[method].append(
                sklearn.metrics.precision_score(
                    true, pred, average=method, labels=labels
                )
            )
            recalls[method].append(
                sklearn.metrics.recall_score(
                    true, pred, average=method, labels=labels
                )
            )
        num_solved += torch.all(true == pred)

    print(f"Num wrong shape predictions: {num_wrong_shape_predictions}")
    print(
        f"Num solved puzzles {num_solved} / {len(labels_true) + num_wrong_shape_predictions}"
    )
    for method in metrics_average_methods:
        mean_precision = torch.tensor(precisions[method]).mean()
        std_precision = torch.tensor(precisions[method]).std()
        mean_recall = torch.tensor(recalls[method]).mean()
        std_recall = torch.tensor(recalls[method]).std()
        print(
            f"Puzzle average precision {method}: {mean_precision:.2f} ± {std_precision:.2f}"
        )
        print(
            f"Puzzle average recall {method}: {mean_recall:.2f} ± {std_recall:.2f}"
        )

    # Compute metrics over all puzzles
    labels_true = torch.hstack(labels_true)
    labels_predicted = torch.hstack(labels_predicted)
    for method in metrics_average_methods:
        print(
            f"Precison {method}: {sklearn.metrics.precision_score(labels_true, labels_predicted, average=method, labels=labels)}"
        )
        print(
            f"Recall {method}: {sklearn.metrics.recall_score(labels_true, labels_predicted, average=method, labels=labels)}"
        )
    print(
        sklearn.metrics.confusion_matrix(
            labels_true, labels_predicted, labels=labels
        )
    )


if __name__ == "__main__":
    main()
