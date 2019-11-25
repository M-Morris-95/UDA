# UDA
My attempt at recreating the experiments in the paper Unsupervised Data Augmentation for Consistency Training.

With the training set split 5% labelled and 95% unlabelled the unsupervised loss was able to leverage the unlabelled data
to improve accuracy from 80.4% to 88%.

The training is sensitive to the parameter lambda which weights the unsupervised loss. When this is too high the network will
find a local optima by setting all outputs to be equal, thus minimising the difference between normal and augmented training samples.

RandAugment is taken from the https://github.com/google-research/uda


@article{xie2019unsupervised,
  title={Unsupervised Data Augmentation for Consistency Training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1904.12848},
  year={2019}
}

@article{cubuk2019randaugment,
  title={RandAugment: Practical data augmentation with no separate search},
  author={Cubuk, Ekin D and Zoph, Barret and Shlens, Jonathon and Le, Quoc V},
  journal={arXiv preprint arXiv:1909.13719},
  year={2019}
}
