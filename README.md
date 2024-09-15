# ML attempt solution for [ARC 2024](https://www.kaggle.com/competitions/arc-prize-2024/overview) Kaggle challange

This is an attempt to solve the ARC 2024 problem using ML. Sadly, it does not
work :(.

# The problem

The ARC challenge is to solve a series of visual-logic puzzles. You are given
a couple of example inputs and outputs. You must deduce the pattern and apply
it to the test input and generate the solution.

The puzzles are all unique and they range from simple puzzles like horizontal/vertical
flip of the input to very [complex ones](https://arcprize.org/play?task=009d5c81)
like coloring an irregular shape with color given a shape that needs to be
deduced across examples. See the [ARC Prize](https://arcprize.org) for more
examples and information.

A problem is considered solved if the output is exact.

# The data

The public data is split into 2 sets based on the level of difficulty. Each
set consists of 400 unique puzzles. The values in the puzzle images are in the
range [0, 9] and the size of the image (input/output) ranges in [1, 30].

Given the limited amount of data training an ML model for such complex task is
a challenge. In order to give the model to learn as much as possible we can try
to augment the dataset through various transformation. Given the visual aspect
of the puzzles computer vision data augmentation techniques are useful but most
of them would invalidate the puzzle. This is especially true for the difficult
sub set.

In order to preserver as much as possible of the puzzles logic I've selected the
following subset of transformations:
* Multiple rotations of 90 degrees - applied to individual example I/O pairs
* Horizontal and vertical flips - applied to individual example I/O pairs
* Value permutation (if the values are permuted the puzzle should hold) - applied to all I/O example pairs to cover some of the puzzles where the pattern is across examples
* Re-sampling and shuffling the I/O examples

Applying these techniques I've generated an augmented dataset of 400,000 puzzles
which does trade off some of the puzzles validity for more samples.

# The model

The model must address the following challenges:
* support arbitrary number of input examples
* support arbitrary shapes for the example images
* generate arbitrary shape output

Because the puzzles have a "sequential" aspect to them - given these examples
predict what follows an autoregressive model seems most natural and given the
success of Transformer architecture in NLP this is what We've selected.

In order to make the input conform to the needs of the Transformer architecture
and address the above challenges We've employed the following input/output
encoding/decoding scheme.

## Encoding

Each image is transformed into tokens using run length encoding for each row.
This gives results in 30 possible tokens for each value. Between the tokens of
each row a special token, `<rhs>`, is inserted in order to introduce information
about the image width.

```python
image = [[0, 0, 0, 0],
         [1, 2, 3, 3]]
encoding = ['<0run4>', '<rhs>', '<1run1>', '<2run1>', '<3run2>', '<rhs>']
```

After each image is encoded using the run length based scheme. The resulting
sequences are concatenated and additional special tokens are inserted in order
to mark the beginning and end of input and output pairs. These tokens add
information about the height of the image and which part of the example it is.

```python input = [[0, 0, 0, 0],
         [1, 2, 3, 3]]
output = [[1]]
encoding = ['<boi>', '<0run4>', '<rhs>', '<1run1>', '<2run1>', '<3run2>', '<rhs>', '<eoi>', '<boo>', '<1run1>', '<rhs>', '<eoo>']
```

## Generation and decoding

To generate the solution to a puzzle first the examples and the input are encoded
and the begin of output, `<boo>`, token is appended. For generation strategy
any of the common text generation strategies, greedy search, contrastive search, etc.
could be used. I've selected greedy for its simplicity.

To further help the model to generate the right shape we customize the generation
for the case when we can infer the output shape. In order to infer the output shape
the ration of of input and output shape is computed and if it's the same than
it is assumed that the test input and solution must have the same ratio. In this
case tokens are generated one by one to fill each row up to the expected width.
Any special tokens are ignored and the `<rhs>` token is inserted when the row
is completed. For the case when the token with the highest probability would
exceed the expected width when decoded, the generated token is replaced with a
token of the same value but with run to match the remaining pixels. E.g. if there
are only 2 pixels left and an `<1run10>` token is generated, it is replaced with
a `<1run2>` token.

For the free generation case (when we can't infer the output shape) we generate
tokens until `<eoo>` is generated and it's constraint to the maximum shape of 30x30.

## Generation of a solution

To generate the solution to a puzzle a Monte Carlo like sampling is employed.
Because, **for some puzzles at least**, augmenting the examples keeps the puzzle
pattern allows us to sample multiple solutions. To get the solution from the
sampled solution the per pixel mode is compute to get the predicted value.

For the free generation case the minimum shape is used as the solution shape.

## Model training

The model is an Decoder only Transformer that is trained with next token prediction loss.

Because the `0` value is used as background value and is most common value and
because during augmentation with value permutation some values will get more
representation the dataset label distribution is imbalanced and we correct it
using label weights computed across full augmented dataset.

The dataset is split into train/test split with 75/25 proportions and stratified
based on problem id.

The optimizer used is AdamW with default Pytorch parameters.

The learning rate is constant `1e-4`.

For regularization Dropout in transformer blocks and weight decay are used.

The model is trained for 10 epochs for about 24h on a NVIDIA RTX3070 GPU which
reaches a mean cross entropy loss of about 0.45 on train and 0.48 on test.

The trained model is available [here](https://www.kaggle.com/models/cosminalexandru/arc_transformer/) in both
pytorch and pythorch-lightning checkpoints mode.

# Evaluation
We compute the precision and recall for the predicted pixel values at a puzzle
level as well as overall.


<table>
<caption>Pixel classification metrics</caption>
<thead>
<tr>
    <th colspan="1"></th>
    <th colspan="1"></th>
    <th colspan="2">Per puzzle precision mean/std</th>
    <th colspan="2">Per puzzle recall mean/std</th>
    <th colspan="2">Overall precision</th>
    <th colspan="2">Overall recall</th>
    <th colspan="1"></th>
    <th colspan="1"></th>
    <th colspan="1"></th>
</tr>
<tr>
    <th>Puzzle set</th>
    <th>Num MC samples</th>
    <th>macro</th>
    <th>weighted</th>
    <th>macro</th>
    <th>weighted</th>
    <th>macro</th>
    <th>weighted</th>
    <th>macro</th>
    <th>weighted</th>
    <th>Num wrong shapes</th>
    <th>Num solved/total</th>
    <th>Inference time on NVIDIA RTX3070</th>
</tr>
</thead>
<tbody>
<tr>
    <td>Easy</td>
    <td>10</td>
    <td>0.19 ± 0.13</td>
    <td>0.66 ± 0.26</td>
    <td>0.18 ± 0.12</td>
    <td>0.68 ± 0.26</td>
    <td>0.56</td>
    <td>0.68</td>
    <td>0.51</td>
    <td>0.70</td>
    <td>50</td>
    <td>25/416</td>
    <td>1h3min</td>
</tr>
<tr>
    <td>Easy</td>
    <td>50</td>
    <td>0.19 ± 0.13</td>
    <td>0.67 ± 0.26</td>
    <td>0.18 ± 0.12</td>
    <td>0.69 ± 0.25</td>
    <td>0.56</td>
    <td>0.69</td>
    <td>0.52</td>
    <td>0.70</td>
    <td>50</td>
    <td>30/416</td>
    <td>5h17min</td>
</tr>
<tr>
    <td>Hard</td>
    <td>10</td>
    <td>0.20 ± 0.15</td>
    <td>0.65 ± 0.23</td>
    <td>0.18 ± 0.12</td>
    <td>0.68 ± 0.22</td>
    <td>0.53</td>
    <td>0.65</td>
    <td>0.44</td>
    <td>0.67</td>
    <td>58</td>
    <td>10/419</td>
    <td>2h9min</td>
</tr>
</tbody>
</table>

# Conclusions

Because the the easy set puzzles are more amenable to being modeled statistically
the model is able to solve more problems from the easy set than from the hard.
Also, on the hard set there are more puzzles for which the shape match is wrong
which idicates that the models has some difficulty in learning to predict output
boundries.

On the hidden test set the model is unable to solve any of the puzzles. A possible
approch to try and improve performance on the hidden test set could be to include
in the prediction step a fintuning step on the examples.

Compute resources requirements are somewhat high for both training and inference.

Given the rellativley good results in predicting the per pixel value it might
be interesting to apply the model to computer vision problem where the challanges
from ARC puzzles almost dissapear:
* Abundance of data and easy to augment without any trade offs
* Constant image shape can be used to ease the burden on the model to learn image shape.
* Computer vision problems do not require exact predictions
