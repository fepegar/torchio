# [TorchIO](http://torchio.rtfd.io/)

> *Tools like TorchIO are a symptom of the maturation of medical AI research using deep learning techniques*.

Jack Clark, Policy Director
at [OpenAI](https://openai.com/) ([link](https://jack-clark.net/2020/03/17/)).

---

<table align="center">
    <tr>
        <td align="left">
            <b>Package</b>
        </td>
        <td align="center">
            <a href="https://pypi.org/project/torchio/">
                <img src="https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white" alt="PyPI downloads">
            </a>
            <a href="https://pypi.org/project/torchio/">
                <img src="https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white" alt="PyPI version">
            </a>
            <a href="https://github.com/fepegar/torchio#contributors-">
                <img src="https://img.shields.io/badge/All_contributors-12-orange.svg" alt="All Contributors">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Docs</b>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/?badge=latest">
                <img src="https://img.shields.io/readthedocs/torchio?label=Docs&logo=Read%20the%20Docs" alt="Documentation status">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Build</b>
        </td>
        <td align="center">
            <a href="https://travis-ci.org/fepegar/torchio">
                <img src="https://img.shields.io/travis/fepegar/torchio/master.svg?label=Travis%20CI%20build&logo=travis" alt="Build status">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Coverage</b>
        </td>
        <td align="center">
            <a href="https://codecov.io/github/fepegar/torchio">
                <img src="https://codecov.io/gh/fepegar/torchio/branch/master/graphs/badge.svg" alt="Coverage status">
            </a>
            <a href='https://coveralls.io/github/fepegar/torchio?branch=master'>
                <img src='https://coveralls.io/repos/github/fepegar/torchio/badge.svg?branch=master' alt='Coverage Status' />
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Code</b>
        </td>
        <td align="center">
            <a href="https://scrutinizer-ci.com/g/fepegar/torchio/?branch=master">
                <img src="https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer" alt="Code quality">
            </a>
            <a href="https://codeclimate.com/github/fepegar/torchio/maintainability">
                <img src="https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability" alt="Code maintainability">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Notebook</b>
        </td>
        <td align="center">
            <a href="https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Social</b>
        </td>
        <td align="center">
            <a href="https://join.slack.com/t/torchioworkspace/shared_invite/zt-exgpd5rm-BTpxg2MazwiiMDw7X9xMFg">
                <img src="https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack" alt="Slack">
            </a>
        </td>
    </tr>
</table>

---

<table align="center">
    <tr>
        <td align="center">Original</td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomblur">Random blur</a>
        </td>
    </tr>
    <tr>
        <td align="center"><img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/1_Lambda_mri.png" alt="Original"></td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomblur">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/2_RandomBlur_mri.gif" alt="Random blur">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomflip">Random flip</a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomnoise">Random noise</a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomflip">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/3_RandomFlip_mri.gif" alt="Random flip">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomnoise">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/4_Compose_mri.gif" alt="Random noise">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomaffine">Random affine transformation</a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomelasticdeformation">Random elastic transformation</a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomaffine">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/5_RandomAffine_mri.gif" alt="Random affine transformation">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomelasticdeformation">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/6_RandomElasticDeformation_mri.gif" alt="Random elastic transformation">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randombiasfield">Random bias field artifact</a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randommotion">Random motion artifact</a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randombiasfield">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/7_RandomBiasField_mri.gif" alt="Random bias field artifact">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randommotion">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/8_RandomMotion_mri.gif" alt="Random motion artifact">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomspike">Random spike artifact</a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomghosting">Random ghosting artifact</a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomspike">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/9_RandomSpike_mri.gif" alt="Random spike artifact">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomghosting">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/gifs_readme/10_RandomGhosting_mri.gif" alt="Random ghosting artifact">
            </a>
        </td>
    </tr>
</table>

---



<a href="https://torchio.readthedocs.io/data/patch_based.html">
    <img align="center" style="width: 640px; height: 360px; overflow: hidden;" src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/queue.gif" alt="Queue">
</a>

([Queue](https://torchio.readthedocs.io/data/patch_training.html#queue)
for [patch-based training](https://torchio.readthedocs.io/data/patch_based.html))

---

TorchIO is a Python package containing a set of tools to efficiently
read, preprocess, sample, augment, and write 3D medical images in deep learning applications
written in [PyTorch](https://pytorch.org/),
including intensity and spatial transforms
for data augmentation and preprocessing.
Transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
[MRI magnetic field inhomogeneity](http://mriquestions.com/why-homogeneity.html)
or [k-space motion artifacts](http://proceedings.mlr.press/v102/shaw19a.html).

This package has been greatly inspired by NiftyNet,
[which is not actively maintained anymore](https://github.com/NifTK/NiftyNet/commit/935bf4334cd00fa9f9d50f6a95ddcbfdde4031e0).


## Credits

If you like this repository, please click on Star!

If you use this package for your research, please cite the paper:

[Pérez-García et al., 2020, *TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning*](https://arxiv.org/abs/2003.04696).


BibTeX entry:

```bibtex
@article{perez-garcia_torchio_2020,
    title = {{TorchIO}: a {Python} library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
    shorttitle = {{TorchIO}},
    url = {http://arxiv.org/abs/2003.04696},
    urldate = {2020-03-11},
    journal = {arXiv:2003.04696 [cs, eess, stat]},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, Sebastien},
    month = mar,
    year = {2020},
    note = {arXiv: 2003.04696},
    keywords = {Computer Science - Computer Vision and Pattern Recognition, Electrical Engineering and Systems Science - Image and Video Processing, Computer Science - Machine Learning, Computer Science - Artificial Intelligence, Statistics - Machine Learning},
}
```


## Getting started

See [Getting started](https://torchio.readthedocs.io/quickstart.html) for
[installation](https://torchio.readthedocs.io/quickstart.html#installation)
instructions,
a [Hello, World!](https://torchio.readthedocs.io/quickstart.html#hello-world)
example and a comprehensive Jupyter notebook tutorial on
[Google Colab](https://torchio.readthedocs.io/quickstart.html#google-colab-jupyter-notebok).

All the documentation is hosted on
[Read the Docs](http://torchio.rtfd.io/).

Please
[open a new issue](https://github.com/fepegar/torchio/issues/new/choose)
if you think something is missing.


## Contributors

Thanks goes to all these people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/fepegar"><img src="https://avatars1.githubusercontent.com/u/12688084?v=4" width="100px;" alt=""/><br /><sub><b>Fernando Pérez-García</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=fepegar" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/commits?author=fepegar" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/romainVala"><img src="https://avatars1.githubusercontent.com/u/5611962?v=4" width="100px;" alt=""/><br /><sub><b>valabregue</b></sub></a><br /><a href="#ideas-romainVala" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/fepegar/torchio/pulls?q=is%3Apr+reviewed-by%3AromainVala" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/fepegar/torchio/commits?author=romainVala" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/GFabien"><img src="https://avatars1.githubusercontent.com/u/39873986?v=4" width="100px;" alt=""/><br /><sub><b>GFabien</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=GFabien" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/GReguig"><img src="https://avatars1.githubusercontent.com/u/11228281?v=4" width="100px;" alt=""/><br /><sub><b>G.Reguig</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=GReguig" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nwschurink"><img src="https://avatars3.githubusercontent.com/u/12720130?v=4" width="100px;" alt=""/><br /><sub><b>Niels Schurink</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=nwschurink" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/ibrhad/ "><img src="https://avatars3.githubusercontent.com/u/18015788?v=4" width="100px;" alt=""/><br /><sub><b>Ibrahim Hadzic</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Aibro45" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/ReubenDo"><img src="https://avatars1.githubusercontent.com/u/17268715?v=4" width="100px;" alt=""/><br /><sub><b>ReubenDo</b></sub></a><br /><a href="#ideas-ReubenDo" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://julianklug.com"><img src="https://avatars0.githubusercontent.com/u/8020367?v=4" width="100px;" alt=""/><br /><sub><b>Julian Klug</b></sub></a><br /><a href="#ideas-MonsieurWave" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/dvolgyes"><img src="https://avatars1.githubusercontent.com/u/425560?v=4" width="100px;" alt=""/><br /><sub><b>David Völgyes</b></sub></a><br /><a href="#ideas-dvolgyes" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/jfillionr/"><img src="https://avatars0.githubusercontent.com/u/219043?v=4" width="100px;" alt=""/><br /><sub><b>Jean-Christophe Fillion-Robin</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=jcfr" title="Documentation">📖</a></td>
    <td align="center"><a href="http://surajpai.tech"><img src="https://avatars1.githubusercontent.com/u/10467804?v=4" width="100px;" alt=""/><br /><sub><b>Suraj Pai</b></sub></a><br /><a href="#ideas-surajpaib" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/bcdarwin"><img src="https://avatars2.githubusercontent.com/u/164148?v=4" width="100px;" alt=""/><br /><sub><b>Ben Darwin</b></sub></a><br /><a href="#ideas-bcdarwin" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the
[all-contributors](https://github.com/all-contributors/all-contributors)
specification. Contributions of any kind welcome!
