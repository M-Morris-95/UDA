# UDA
My attempt at recreating the experiments in the paper Unsupervised Data Augmentation for Consistency Training.

With the training set split 5% labelled and 95% unlabelled the unsupervised loss was able to leverage the unlabelled data
to improve accuracy from 80.4% to 88%.

The trainign is sensitive to the parameter lambda which weights the unsupervised loss. When this is too high the netwrok will
find a local optima by setting all outputs to be equal, thus minimising the difference between normal and augmented training samples.

