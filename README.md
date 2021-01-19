# Visual SmileyNet


Visual SmileyNet is a library for training an image to emoji neural network model. It contains functionality used to produce the results in the paper:

[Smile, Be Happy :) Emoji Embedding for Visual Sentiment Analysis](https://arxiv.org/abs/1907.06160)
Z. Al-Halah, A. Aitken, W. Shi, J. Caballero, *ICCV CroMoL workshop, 2019*

For more information on this work please visit the [project page](https://www.cs.utexas.edu/~ziad/emoji_visual_sentiment.html).

If you use this code please reference the publication above.

## Installation

All requirements can be install by running:
`python setup.py install`

### Requirements

With python=3.7:

	torch==1.2.0
	torchvision==0.4.0
	numpy==1.16.6
	pillow==6.2.2
	pandas==0.24.0
	requests==2.22.0

## Visual Smiley Dataset

You can download the visual smiley dataset used in this work from [here](https://twitter.app.box.com/v/visual-smiley-dataset).


## Getting Started

### Training

The script `train_model.sh` will train a network to perform image to emoji predictions. The script `train_model_toy.sh` will run the training of a model on a very small subset of the dataset. This can be used to get familiar with the training script before running on the full dataset.

To start training a model on the Visual SmileyNet dataset you can follow these steps:

1. Download the Visual SmileyNet dataset from [here](https://twitter.app.box.com/v/visual-smiley-dataset).
2. Extract the dataset and save the folder location in environment variable DATA_DIR with `export DATA_DIR=</dataset/location/Visual Smiley Dataset>`
3. Run the training script with `sh sh_scripts/train_model.sh`. Alternatively, running `sh sh_scripts/train_model_toy.sh` will run on small subset of the dataset.

### Testing

The script `test_model.sh` will test the model on a set of images.

## Cite us

	@inproceedings{visualsent_iccv_cromol2019,
	    title={Smile, Be Happy :) Emoji Embedding for Visual Sentiment Analysis},
	    author={Ziad Al-Halah and Andrew Aitken and Wenzhe Shi and Jose Caballero},
	    booktitle={ICCV 2019 Workshop on Cross-Modal Learning in Real World},
	    year={2019}
	}

## Security Issues?
Please report sensitive security issues via Twitter's bug-bounty program (https://hackerone.com/twitter) rather than GitHub.
