"""ARC dataset tests"""

import json
import os
import tempfile

import pytest

import arc.dataset


@pytest.fixture()
def train_set():
    problems = {
        "a0": {
            "test": [{"input": [[0, 0], [0, 0]]}],
            "train": [
                {"input": [[1, 1], [1, 1]], "output": [[1, 1], [1, 1]]},
                {"input": [[2, 2], [2, 2]], "output": [[2, 2], [2, 2]]},
            ],
        },
        "a1": {
            "test": [{"input": [[0, 0], [0, 0]]}, {"input": [[1, 1], [1, 1]]}],
            "train": [
                {
                    "input": [
                        [
                            2,
                            2,
                        ],
                        [2, 2],
                    ],
                    "output": [[3, 3], [3, 3]],
                }
            ],
        },
    }
    solutions = {
        "a0": [[[3, 3], [3, 3]]],
        "a1": [[[3, 3], [3, 3]], [[4, 4], [4, 4]]],
    }

    return problems, solutions


@pytest.fixture()
def test_set():
    problems = {
        "b0": {
            "test": [{"input": [[0, 0], [0, 0]]}],
            "train": [
                {"input": [[1, 1], [1, 1]], "output": [[1, 1], [1, 1]]},
                {"input": [[2, 2], [2, 2]], "output": [[2, 2], [2, 2]]},
            ],
        }
    }

    return problems


def test_arc_dataset_construction(train_set, test_set) -> None:
    train_problem, train_solution = train_set
    with tempfile.TemporaryDirectory() as workdir:
        train_problem_path = os.path.join(workdir, "train_problem.json")
        with open(train_problem_path, "w") as problemf:
            json.dump(train_problem, problemf)

        train_solution_path = os.path.join(workdir, "train_solution.json")
        with open(train_solution_path, "w") as solutionf:
            json.dump(train_solution, solutionf)

        test_problem_path = os.path.join(workdir, "test_problem.json")
        with open(test_problem_path, "w") as testf:
            json.dump(test_set, testf)

        dataset = arc.dataset.ARCDataset(
            [
                (train_problem_path, train_solution_path),
                (test_problem_path, None),
            ]
        )

    assert len(dataset) == 3
    assert set(dataset.problems_ids) == set(train_problem) | set(test_set)

    assert len(dataset.problems["a0"]["test"]) == 1
    assert len(dataset.problems["a1"]["test"]) == 2
    assert len(dataset.problems["b0"]["test"]) == 1

    assert (
        dataset.problems["a1"]["test"][0]["output"] == train_solution["a1"][0]
    )
    assert (
        dataset.problems["a1"]["test"][1]["output"] == train_solution["a1"][1]
    )
