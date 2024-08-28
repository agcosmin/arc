"""ARC dataset tests"""

import pytest
import torch

import arc.tokenizer


def test_tokenizer_constructor_raise_for_invalid_max_length():
    with pytest.raises(ValueError):
        arc.tokenizer.ARCTokenizer(-10)


def test_rle_tokenization_returns_expected_tokens_for_single_element():
    tokenizer = arc.tokenizer.ARCTokenizer(max_run_length=3)
    tokens = tokenizer._rle_tokenize(torch.zeros((1,)))
    assert tokens == torch.tensor([len(tokenizer.special_tokens)])

    tokens = tokenizer._rle_tokenize(torch.zeros((1, 1)))
    assert tokens == torch.tensor([len(tokenizer.special_tokens)])


def test_rle_tokenization_returns_expected_tokens_for_multiple_elements():
    max_run_length = 3
    tokenizer = arc.tokenizer.ARCTokenizer(max_run_length=max_run_length)
    tokens = tokenizer._rle_tokenize(torch.arange(4))
    num_special_tokens = len(tokenizer.special_tokens)
    expected_tokens = torch.tensor(
        [
            num_special_tokens + max_run_length * 0,
            num_special_tokens + max_run_length * 1,
            num_special_tokens + max_run_length * 2,
            num_special_tokens + max_run_length * 3,
        ]
    )
    assert torch.all(tokens == expected_tokens)

    tokens = tokenizer._rle_tokenize(torch.arange(4).reshape(2, 2))
    assert torch.all(tokens == expected_tokens)


def test_rle_tokenization_returns_expected_tokens_for_all_run_length():
    max_run_length = 3
    tokenizer = arc.tokenizer.ARCTokenizer(max_run_length=max_run_length)
    tokens = tokenizer._rle_tokenize(
        torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    )
    num_special_tokens = len(tokenizer.special_tokens)
    expected_tokens = torch.tensor(
        [
            num_special_tokens + max_run_length * 0,
            num_special_tokens + max_run_length * 1 + 1,
            num_special_tokens + max_run_length * 2 + 2,
            num_special_tokens + max_run_length * 3 + 2,
            num_special_tokens + max_run_length * 3,
        ]
    )
    assert torch.all(tokens == expected_tokens)


def test_rle_tokenization_returns_expected_tokens_for_rhs():
    max_run_length = 3
    tokenizer = arc.tokenizer.ARCTokenizer(max_run_length)
    tokens = tokenizer._rle_tokenize(
        torch.tensor([[0, 1, 1], [1, 1, 2]]), add_rhs_token=True
    )
    num_special_tokens = len(tokenizer.special_tokens)
    expected_tokens = torch.tensor(
        [
            num_special_tokens + max_run_length * 0 + 0,
            num_special_tokens + max_run_length * 1 + 1,
            tokenizer.special_tokens["<rhs>"],
            num_special_tokens + max_run_length * 1 + 1,
            num_special_tokens + max_run_length * 2 + 0,
        ]
    )
    assert torch.all(tokens == expected_tokens)


def test_rle_problem_encoding_returns_expected_tokens():
    example1 = torch.tensor([[0, 0, 0], [0, 0, 0]])
    solution1 = torch.tensor([[1, 1], [1, 1]])
    example2 = torch.tensor([[2, 2, 2, 2], [0, 1, 2, 3]])
    problem = [example1, solution1, example2]

    max_run_length = 3
    tokenizer = arc.tokenizer.ARCTokenizer(max_run_length)
    num_special_tokens = len(tokenizer.special_tokens)
    expected_tokens = torch.tensor(
        [
            tokenizer.special_tokens["<in>"],
            num_special_tokens + max_run_length * 0 + 2,
            tokenizer.special_tokens["<rhs>"],
            num_special_tokens + max_run_length * 0 + 2,
            tokenizer.special_tokens["<out>"],
            num_special_tokens + max_run_length * 1 + 1,
            tokenizer.special_tokens["<rhs>"],
            num_special_tokens + max_run_length * 1 + 1,
            tokenizer.special_tokens["<in>"],
            num_special_tokens + max_run_length * 2 + 2,
            num_special_tokens + max_run_length * 2 + 0,
            tokenizer.special_tokens["<rhs>"],
            num_special_tokens + max_run_length * 0 + 0,
            num_special_tokens + max_run_length * 1 + 0,
            num_special_tokens + max_run_length * 2 + 0,
            num_special_tokens + max_run_length * 3 + 0,
            tokenizer.special_tokens["<out>"],
        ]
    )

    tokens = tokenizer.encode(problem, add_special_tokens=True)
    assert torch.all(tokens == expected_tokens)


def test_decode_encoded_sequence():
    example1 = torch.tensor([[0, 0, 1, 2, 2], [1, 1, 1, 0, 0], [0, 1, 2, 1, 2]])
    solution1 = torch.tensor([[1, 1], [0, 2]])
    example2 = torch.tensor([[1, 1, 1, 2, 2], [0, 0, 0, 0, 0], [1, 1, 2, 1, 2]])
    solution2 = torch.tensor([[1, 2], [0, 2]])
    sequence = [example1, solution1, example2, solution2]
    tokenizer = arc.tokenizer.ARCTokenizer(max_run_length=3)

    tokenized_sequence = tokenizer.encode(sequence, add_solution_prompt=True)
    decoded_sequence = tokenizer.decode(tokenized_sequence)
    for decoded, expected in zip(decoded_sequence, sequence):
        assert torch.all(decoded == expected)

    tokenized_sequence = tokenizer.encode(sequence, add_solution_prompt=False)
    decoded_sequence = tokenizer.decode(tokenized_sequence)
    for decoded, expected in zip(decoded_sequence, sequence):
        assert torch.all(decoded == expected)
