<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU GPL-v3.0][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://SynthRAD2025.grand-challenge.org/">
    <img src="./SynthRAD_banner.png" alt="Logo" width="770" height="160">
  </a>


  <p align="center">
    Preparing the metrics for evaluation of 
<a href="https://SynthRAD2025.grand-challenge.org/"><strong>SynthRAD2025 Grand Challenge</strong></a>
  <br />
    <a href="https://github.com/SynthRAD2025/metrics"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SynthRAD2025/metrics">View Demo</a>
    ·
    <a href="https://github.com/SynthRAD2025/metrics/issues">Report Bug</a>
    ·
    <a href="https://github.com/SynthRAD2025/metrics/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Goal](#goal)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Function Descriptions](#functions-descriptions)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
<!--
* [Acknowledgements](#acknowledgements)
-->


<!-- ABOUT THE PROJECT -->
## Goal

Assess the quality of the synthetic computed tomography (sCT) images
against CT. A brief description of the metrics can be found at: [https://synthrad2025.grand-challenge.org/metrics-ranking/](https://synthrad2025.grand-challenge.org/metrics-ranking/).

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.


### Installation

1. Clone the repo
```sh
git clone https://github.com/SynthRAD2025/metrics.git
```
or
```sh
git clone git@github.com:SynthRAD2025/metrics.git
```

### Prerequisites

* numpy
```sh
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->

## Usage

The metrics are computed in two files: `image_metrics.py` and `dose_metrics.py`.
These compute respectively,
* The image similarity between the ground-truth CT and the synthetic CT. Thes metrics include the mean squared error (MSE), peak signal to noise ratio (PSNR), and multiscale structural similarity (MS-SSIM).
* The segmentation metrics evaluate the geometric accuracy between the ground-truth CT and synthetic CT. They compute the DICE and 95th percentile Hausdorff distance (HD95) between a ground-truth segmentation set and segmentations estimated from synthetic CTs. The weights and implementations of the TotalSegmentator model are archived and illustrated in the [evaluation](https://github.com/SynthRAD2025/evaluation/) repository.
* The metrics to compare the dose delivered to the ground truth and the synthetic CT. These metrics include the mean absolute dose (MAE), a dose-volume histogram (DVH) metric, and the gamma pass rate. 



### Functions Descriptions
In general, any function can be used in the following way.

**a(input, output)**

	description:
	compute the metric a (e.g., mse, psnr, ssim) between input and output
	
	arguments:
	input: The numpy array of the ground-truth image
	output: The numpy array of the predicted image

All metrics can be computed by using the `score_patient`, which loads the data and returns all metrics:

**Image similarity**
``` 
    metrics = ImageMetrics()
    ground_truth_path = "path/to/ground_truth.mha"
    predicted_path = "path/to/prediction.mha"
    mask_path = "path/to/mask.mha"
    print(metrics.score_patient(ground_truth_path, predicted_path, mask_path))
```

**Geometry consistency**
``` 
    metrics = SegmentationMetrics()
    ground_truth_path = "path/to/ground_truth.mha"
    predicted_path = "path/to/prediction.mha"
    mask_path = "path/to/mask.mha"
    print(metrics.score_patient(ground_truth_path, predicted_path, mask_path))
```

**Dose evaluation**
``` 
    dose_path = 'path/to/treatment_plans'
    predicted_path = "path/to/prediction.mha"
    patient_id="1BA000"
    
    metrics = DoseMetrics(dose_path)
    print(metrics.score_patient(patient_id, predicted_path))
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/SynthRAD2025/metrics/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Maarten Terpstra - M.L.Terpstra-5@umcutrecht.nl
Matteo Maspero - [@matteomasperonl](https://bsky.app/profile/matteomaspero.bsky.social) - m.maspero@umcutrecht.nl

Project Link: [https://github.com/SynthRAD2025/metrics](https://github.com/SynthRAD2025/metrics)


<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* []()
* []()
* []()
-->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/
#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SynthRAD2025/repo.svg?style=flat-square
[contributors-url]: https://github.com/SynthRAD2025/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SynthRAD2025/repo.svg?style=flat-square
[forks-url]: https://github.com/SynthRAD2025/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/SynthRAD2025/repo.svg?style=flat-square
[stars-url]: https://github.com/SynthRAD2025/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/SynthRAD2025/repo.svg?style=flat-square
[issues-url]: https://github.com/SynthRAD2025/repo/issues
[license-shield]: https://img.shields.io/github/license/SynthRAD2025/repo.svg?style=flat-square
[license-url]: https://github.com/SynthRAD2025/repo/blob/master/LICENSE.txt
