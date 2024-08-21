# ili-summarizer

This is a python package which consolidates tools for constructing summary statistics from spectroscopic surveys of the universe's Large-Scale Structure (LSS). It is built to support the [Learning the Universe CMASS (ltu-cmass) pipeline](https://github.com/maho3/ltu-cmass). 

## Fuctionality
Currently, ili-summarizer supports summary statistic calculation with halo count fields. The available statistics include:
* Two-point statistics
    * [Two point correlation function](summarizer/two_point/corr.py)
    * [Power spectrum](summarizer/two_point/Pk.py)
    * [Marked power spectrum](summarizer/two_point/marked_Pk.py)
* Three-point statistics
    * [Bispectrum](summarizer/three_point/Bk.py)
* Wavelet statistics
    * [Wavelet scattering transform coefficients](summarizer/wavelet/wst.py)
* Nearest Neighbor statistics
    * [kNN](summarizer/knn/knn.py)
* Environment Satistics
    * [Density Split](summarizer/environment/density_split.py)
* Void Statistics
    * [VPF](summarizer/vpf/vpf.py)
    * [CiC](summarizer/cic/cic.py)


## Installing 
Follow the instructions detailed in [INSTALL.md](INSTALL.md).

## Contact
If you have comments, questions, or feedback, please [write us an issue](https://github.com/florpi/ili-summarizer/issues). You can also contact Carolina Cuesta (cuestalz@mit.edu) or Matthew Ho (matthew.annam.ho@gmail.com).

## Contributors
Below is a list of contributors to this repository. (Please add your name here!)
* [Carolina Cuesta-Lazaro](https://github.com/florpi) (MIT)
* [Matthew Ho](https://github.com/maho3) (IAP)
* [Deaglan Bartlett](https://github.com/DeaglanBartlett) (IAP)

## Acknowledgements
This work was supported by the [Simons Collaboration on "Learning the Universe"](https://www.learning-the-universe.org/).
