<!-- markdownlint-disable -->
<p align="center">
  <a href="http://torchio.rtfd.io/">
    <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/source/favicon_io/for_readme_2000x462.png" alt="TorchIO logo">
  </a>
</p>
<!-- markdownlint-restore -->

> *Tools like TorchIO are a symptom of the maturation of medical AI research using deep learning techniques*.

Jack Clark, Policy Director
at [OpenAI](https://openai.com/) ([link](https://jack-clark.net/2020/03/17/)).

---

<!-- markdownlint-disable -->
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
            <a href="https://anaconda.org/conda-forge/torchio">
                <img src="https://img.shields.io/conda/v/conda-forge/torchio.svg?label=conda-forge&logo=conda-forge" alt="Conda version">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>CI</b>
        </td>
        <td align="center">
            <a href="https://github.com/fepegar/torchio/actions/workflows/tests.yml">
                <img src="https://github.com/fepegar/torchio/actions/workflows/tests.yml/badge.svg" alt="Tests status">
            </a>
            <a href="https://torchio.rtfd.io/?badge=latest">
                <img src="https://img.shields.io/readthedocs/torchio?label=Docs&logo=Read%20the%20Docs" alt="Documentation status">
            </a>
            <a href="https://codecov.io/github/fepegar/torchio">
                <img src="https://codecov.io/gh/fepegar/torchio/branch/main/graphs/badge.svg" alt="Coverage status">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Code</b>
        </td>
        <td align="center">
            <a href="https://scrutinizer-ci.com/g/fepegar/torchio/?branch=main">
                <img src="https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer" alt="Code quality">
            </a>
            <a href="https://codeclimate.com/github/fepegar/torchio/maintainability">
                <img src="https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability" alt="Code maintainability">
            </a>
            <a href="https://github.com/pre-commit/pre-commit">
                <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Tutorials</b>
        </td>
        <td align="center">
            <a href="https://github.com/fepegar/torchio/blob/main/tutorials/README.md">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Community</b>
        </td>
        <td align="center">
            <a href="https://join.slack.com/t/torchioworkspace/shared_invite/zt-exgpd5rm-BTpxg2MazwiiMDw7X9xMFg">
                <img src="https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack" alt="Slack">
            </a>
            <a href="https://twitter.com/TorchIOLib">
                <img src="https://img.shields.io/twitter/url/https/twitter.com/TorchIOLib.svg?style=social&label=Follow%20%40TorchIOLib" alt="Twitter">
            </a>
            <a href="https://twitter.com/TorchIO_commits">
                <img src="https://img.shields.io/twitter/url/https/twitter.com/TorchIO_commits.svg?style=social&label=Follow%20%40TorchIO_commits" alt="Twitter">
            </a>
            <a href="https://www.youtube.com/watch?v=UEUVSw5-M9M">
                <img src="https://img.shields.io/youtube/views/UEUVSw5-M9M?label=watch&style=social" alt="YouTube">
            </a>
        </td>
    </tr>
</table>

---

<p align="center">
  <a href="https://torchio.readthedocs.io/transforms/augmentation.html">
    <img style="width: 600px; overflow: hidden;" src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/fpg_progressive.gif" alt="Progressive artifacts">
  </a>
</p>

<p align="center">
  <a href="https://torchio.readthedocs.io/transforms/augmentation.html">
    <img style="width: 360px; height: 360px; overflow: hidden;" src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/augmentation.gif" alt="Augmentation">
  </a>
</p>

---

<table align="center">
    <tr>
        <td align="center">Original</td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomblur">Random blur</a>
        </td>
    </tr>
    <tr>
        <td align="center"><img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/1_Lambda_mri.png" alt="Original"></td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomblur">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/2_RandomBlur_mri.gif" alt="Random blur">
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
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/3_RandomFlip_mri.gif" alt="Random flip">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomnoise">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/4_Compose_mri.gif" alt="Random noise">
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
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/5_RandomAffine_mri.gif" alt="Random affine transformation">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomelasticdeformation">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/6_RandomElasticDeformation_mri.gif" alt="Random elastic transformation">
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
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/7_RandomBiasField_mri.gif" alt="Random bias field artifact">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randommotion">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/8_RandomMotion_mri.gif" alt="Random motion artifact">
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
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/9_RandomSpike_mri.gif" alt="Random spike artifact">
            </a>
        </td>
        <td align="center">
            <a href="http://torchio.rtfd.io/transforms/augmentation.html#randomghosting">
                <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/gifs_readme/10_RandomGhosting_mri.gif" alt="Random ghosting artifact">
            </a>
        </td>
    </tr>
</table>

---

<p align="center">
  <a href="https://torchio.readthedocs.io/patches/patch_training.html#queue">
    <img style="width: 640px; height: 360px; overflow: hidden;" src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/images/queue.gif" alt="Queue">
  </a>
</p>

([Queue](https://torchio.readthedocs.io/patches/patch_training.html#queue)
for [patch-based training](https://torchio.readthedocs.io/patches/index.html))

<!-- markdownlint-restore -->

---

TorchIO is a Python package containing a set of tools to efficiently
read, preprocess, sample, augment, and write 3D medical images in deep learning applications
written in [PyTorch](https://pytorch.org/),
including intensity and spatial transforms
for data augmentation and preprocessing.
Transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
[MRI magnetic field inhomogeneity](https://mriquestions.com/why-homogeneity.html)
or [k-space motion artifacts](http://proceedings.mlr.press/v102/shaw19a.html).

This package has been greatly inspired by NiftyNet,
[which is not actively maintained anymore](https://github.com/NifTK/NiftyNet/commit/935bf4334cd00fa9f9d50f6a95ddcbfdde4031e0).

## Credits

If you like this repository, please click on Star!

If you use this package for your research, please cite our paper:

[F. Pérez-García, R. Sparks, and S. Ourselin. *TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning*. Computer Methods and Programs in Biomedicine (June 2021), p. 106236. ISSN: 0169-2607.doi:10.1016/j.cmpb.2021.106236.](https://doi.org/10.1016/j.cmpb.2021.106236)

BibTeX entry:

```bibtex
@article{perez-garcia_torchio_2021,
    title = {TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
    journal = {Computer Methods and Programs in Biomedicine},
    pages = {106236},
    year = {2021},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2021.106236},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260721003102},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, S{\'e}bastien},
}
```

This project is supported by the following institutions:

- [Engineering and Physical Sciences Research Council (EPSRC) & UK Research and Innovation (UKRI)](https://epsrc.ukri.org/)
- [EPSRC Centre for Doctoral Training in Intelligent, Integrated Imaging In Healthcare (i4health)](https://www.ucl.ac.uk/intelligent-imaging-healthcare/) (University College London)
- [Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS)](https://www.ucl.ac.uk/interventional-surgical-sciences/) (University College London)
- [School of Biomedical Engineering & Imaging Sciences (BMEIS)](https://www.kcl.ac.uk/bmeis) (King's College London)

## Getting started

See [Getting started](https://torchio.readthedocs.io/quickstart.html) for
[installation](https://torchio.readthedocs.io/quickstart.html#installation)
instructions
and a [Hello, World!](https://torchio.readthedocs.io/quickstart.html#hello-world)
example.

Longer usage examples can be found in the
[tutorials](https://github.com/fepegar/torchio/blob/main/tutorials/README.md).

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
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fepegar"><img src="https://avatars1.githubusercontent.com/u/12688084?v=4?s=100" width="100px;" alt="Fernando Pérez-García"/><br /><sub><b>Fernando Pérez-García</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=fepegar" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/commits?author=fepegar" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/romainVala"><img src="https://avatars1.githubusercontent.com/u/5611962?v=4?s=100" width="100px;" alt="valabregue"/><br /><sub><b>valabregue</b></sub></a><br /><a href="#ideas-romainVala" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/fepegar/torchio/pulls?q=is%3Apr+reviewed-by%3AromainVala" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/fepegar/torchio/commits?author=romainVala" title="Code">💻</a> <a href="#question-romainVala" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GFabien"><img src="https://avatars1.githubusercontent.com/u/39873986?v=4?s=100" width="100px;" alt="GFabien"/><br /><sub><b>GFabien</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=GFabien" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/pulls?q=is%3Apr+reviewed-by%3AGFabien" title="Reviewed Pull Requests">👀</a> <a href="#ideas-GFabien" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GReguig"><img src="https://avatars1.githubusercontent.com/u/11228281?v=4?s=100" width="100px;" alt="G.Reguig"/><br /><sub><b>G.Reguig</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=GReguig" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nwschurink"><img src="https://avatars3.githubusercontent.com/u/12720130?v=4?s=100" width="100px;" alt="Niels Schurink"/><br /><sub><b>Niels Schurink</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=nwschurink" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/ibrhad/ "><img src="https://avatars3.githubusercontent.com/u/18015788?v=4?s=100" width="100px;" alt="Ibrahim Hadzic"/><br /><sub><b>Ibrahim Hadzic</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Aibro45" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ReubenDo"><img src="https://avatars1.githubusercontent.com/u/17268715?v=4?s=100" width="100px;" alt="ReubenDo"/><br /><sub><b>ReubenDo</b></sub></a><br /><a href="#ideas-ReubenDo" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://julianklug.com"><img src="https://avatars0.githubusercontent.com/u/8020367?v=4?s=100" width="100px;" alt="Julian Klug"/><br /><sub><b>Julian Klug</b></sub></a><br /><a href="#ideas-MonsieurWave" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dvolgyes"><img src="https://avatars1.githubusercontent.com/u/425560?v=4?s=100" width="100px;" alt="David Völgyes"/><br /><sub><b>David Völgyes</b></sub></a><br /><a href="#ideas-dvolgyes" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/fepegar/torchio/commits?author=dvolgyes" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/jfillionr/"><img src="https://avatars0.githubusercontent.com/u/219043?v=4?s=100" width="100px;" alt="Jean-Christophe Fillion-Robin"/><br /><sub><b>Jean-Christophe Fillion-Robin</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=jcfr" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://surajpai.tech"><img src="https://avatars1.githubusercontent.com/u/10467804?v=4?s=100" width="100px;" alt="Suraj Pai"/><br /><sub><b>Suraj Pai</b></sub></a><br /><a href="#ideas-surajpaib" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bcdarwin"><img src="https://avatars2.githubusercontent.com/u/164148?v=4?s=100" width="100px;" alt="Ben Darwin"/><br /><sub><b>Ben Darwin</b></sub></a><br /><a href="#ideas-bcdarwin" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/OeslleLucena"><img src="https://avatars0.githubusercontent.com/u/19919194?v=4?s=100" width="100px;" alt="Oeslle Lucena"/><br /><sub><b>Oeslle Lucena</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3AOeslleLucena" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.soumick.com"><img src="https://avatars0.githubusercontent.com/u/20525305?v=4?s=100" width="100px;" alt="Soumick Chatterjee"/><br /><sub><b>Soumick Chatterjee</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=soumickmj" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/neuronflow"><img src="https://avatars1.githubusercontent.com/u/7048826?v=4?s=100" width="100px;" alt="neuronflow"/><br /><sub><b>neuronflow</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=neuronflow" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jwitos"><img src="https://avatars2.githubusercontent.com/u/948674?v=4?s=100" width="100px;" alt="Jan Witowski"/><br /><sub><b>Jan Witowski</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=jwitos" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dmus"><img src="https://avatars1.githubusercontent.com/u/464378?v=4?s=100" width="100px;" alt="Derk Mus"/><br /><sub><b>Derk Mus</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=dmus" title="Documentation">📖</a> <a href="https://github.com/fepegar/torchio/commits?author=dmus" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Admus" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.linkedin.com/in/che85"><img src="https://avatars2.githubusercontent.com/u/10195822?v=4?s=100" width="100px;" alt="Christian Herz"/><br /><sub><b>Christian Herz</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Ache85" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/efirdc"><img src="https://avatars3.githubusercontent.com/u/5416313?v=4?s=100" width="100px;" alt="Cory Efird"/><br /><sub><b>Cory Efird</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=efirdc" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Aefirdc" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/estebvac"><img src="https://avatars.githubusercontent.com/u/21016728?v=4?s=100" width="100px;" alt="Esteban Vaca C."/><br /><sub><b>Esteban Vaca C.</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Aestebvac" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://stackoverflow.com/users/3250829/rayryeng?tab=profile"><img src="https://avatars.githubusercontent.com/u/765375?v=4?s=100" width="100px;" alt="Ray Phan"/><br /><sub><b>Ray Phan</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Arayryeng" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Linardos"><img src="https://avatars.githubusercontent.com/u/26694607?v=4?s=100" width="100px;" alt="Akis Linardos"/><br /><sub><b>Akis Linardos</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3ALinardos" title="Bug reports">🐛</a> <a href="https://github.com/fepegar/torchio/commits?author=Linardos" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://nina.brown.15@ucl.ac.uk"><img src="https://avatars.githubusercontent.com/u/56116848?v=4?s=100" width="100px;" alt="Nina Montana-Brown"/><br /><sub><b>Nina Montana-Brown</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=NMontanaBrown" title="Documentation">📖</a> <a href="#infra-NMontanaBrown" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fabien-brulport"><img src="https://avatars.githubusercontent.com/u/32873392?v=4?s=100" width="100px;" alt="fabien-brulport"/><br /><sub><b>fabien-brulport</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Afabien-brulport" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/malteekj"><img src="https://avatars.githubusercontent.com/u/44469884?v=4?s=100" width="100px;" alt="malteekj"/><br /><sub><b>malteekj</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Amalteekj" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/diazandr3s"><img src="https://avatars.githubusercontent.com/u/11991079?v=4?s=100" width="100px;" alt="Andres Diaz-Pinto"/><br /><sub><b>Andres Diaz-Pinto</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Adiazandr3s" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cbica.upenn.edu/spati"><img src="https://avatars.githubusercontent.com/u/11719673?v=4?s=100" width="100px;" alt="Sarthak Pati"/><br /><sub><b>Sarthak Pati</b></sub></a><br /><a href="#platform-sarthakpati" title="Packaging/porting to new platform">📦</a> <a href="https://github.com/fepegar/torchio/commits?author=sarthakpati" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GabriellaKamlish"><img src="https://avatars.githubusercontent.com/u/26881445?v=4?s=100" width="100px;" alt="GabriellaKamlish"/><br /><sub><b>GabriellaKamlish</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3AGabriellaKamlish" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TylerSpears"><img src="https://avatars.githubusercontent.com/u/7096950?v=4?s=100" width="100px;" alt="Tyler Spears"/><br /><sub><b>Tyler Spears</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3ATylerSpears" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://dagut.ru"><img src="https://avatars.githubusercontent.com/u/7759336?v=4?s=100" width="100px;" alt="DaGuT"/><br /><sub><b>DaGuT</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=DaGuT" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hsiangyuzhao"><img src="https://avatars.githubusercontent.com/u/53631393?v=4?s=100" width="100px;" alt="Xiangyu Zhao"/><br /><sub><b>Xiangyu Zhao</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Ahsiangyuzhao" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://siahuat0727.github.io"><img src="https://avatars.githubusercontent.com/u/17688111?v=4?s=100" width="100px;" alt="siahuat0727"/><br /><sub><b>siahuat0727</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=siahuat0727" title="Documentation">📖</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Asiahuat0727" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Svdvoort"><img src="https://avatars.githubusercontent.com/u/23049683?v=4?s=100" width="100px;" alt="Svdvoort"/><br /><sub><b>Svdvoort</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=Svdvoort" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/alban-steff-260ab715b/"><img src="https://avatars.githubusercontent.com/u/59876036?v=4?s=100" width="100px;" alt="Albans98"/><br /><sub><b>Albans98</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=Albans98" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://mattwarkentin.github.io/"><img src="https://avatars.githubusercontent.com/u/27825069?v=4?s=100" width="100px;" alt="Matthew T. Warkentin"/><br /><sub><b>Matthew T. Warkentin</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=mattwarkentin" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/glupol"><img src="https://avatars.githubusercontent.com/u/57905234?v=4?s=100" width="100px;" alt="glupol"/><br /><sub><b>glupol</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Aglupol" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ramonemiliani93"><img src="https://avatars.githubusercontent.com/u/14314888?v=4?s=100" width="100px;" alt="ramonemiliani93"/><br /><sub><b>ramonemiliani93</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=ramonemiliani93" title="Documentation">📖</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Aramonemiliani93" title="Bug reports">🐛</a> <a href="https://github.com/fepegar/torchio/commits?author=ramonemiliani93" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/justusschock"><img src="https://avatars.githubusercontent.com/u/12886177?v=4?s=100" width="100px;" alt="Justus Schock"/><br /><sub><b>Justus Schock</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=justusschock" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Ajustusschock" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cepa995"><img src="https://avatars.githubusercontent.com/u/67524891?v=4?s=100" width="100px;" alt="Stefan Milorad Radonjić"/><br /><sub><b>Stefan Milorad Radonjić</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Acepa995" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/srg9000"><img src="https://avatars.githubusercontent.com/u/26834833?v=4?s=100" width="100px;" alt="Sajan Gohil"/><br /><sub><b>Sajan Gohil</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Asrg9000" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://bandism.net/"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt="Ikko Ashimine"/><br /><sub><b>Ikko Ashimine</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=eltociear" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/laynr"><img src="https://avatars.githubusercontent.com/u/775607?v=4?s=100" width="100px;" alt="laynr"/><br /><sub><b>laynr</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=laynr" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/omarespejel"><img src="https://avatars.githubusercontent.com/u/4755430?v=4?s=100" width="100px;" alt="Omar U. Espejel"/><br /><sub><b>Omar U. Espejel</b></sub></a><br /><a href="#audio-omarespejel" title="Audio">🔊</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jamesobutler"><img src="https://avatars.githubusercontent.com/u/15837524?v=4?s=100" width="100px;" alt="James Butler"/><br /><sub><b>James Butler</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Ajamesobutler" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/res191"><img src="https://avatars.githubusercontent.com/u/6549034?v=4?s=100" width="100px;" alt="res191"/><br /><sub><b>res191</b></sub></a><br /><a href="#fundingFinding-res191" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nengwp"><img src="https://avatars.githubusercontent.com/u/44516353?v=4?s=100" width="100px;" alt="nengwp"/><br /><sub><b>nengwp</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Anengwp" title="Bug reports">🐛</a> <a href="https://github.com/fepegar/torchio/commits?author=nengwp" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/susanveraclarke"><img src="https://avatars.githubusercontent.com/u/93313094?v=4?s=100" width="100px;" alt="susanveraclarke"/><br /><sub><b>susanveraclarke</b></sub></a><br /><a href="#design-susanveraclarke" title="Design">🎨</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://nepersica.tistory.com"><img src="https://avatars.githubusercontent.com/u/45097022?v=4?s=100" width="100px;" alt="nepersica"/><br /><sub><b>nepersica</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Anepersica" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://spenhouet.com"><img src="https://avatars.githubusercontent.com/u/7819068?v=4?s=100" width="100px;" alt="Sebastian Penhouet"/><br /><sub><b>Sebastian Penhouet</b></sub></a><br /><a href="#ideas-Spenhouet" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Bigsealion"><img src="https://avatars.githubusercontent.com/u/23148550?v=4?s=100" width="100px;" alt="Bigsealion"/><br /><sub><b>Bigsealion</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3ABigsealion" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.kitware.com/dzenan-zukic/"><img src="https://avatars.githubusercontent.com/u/1792121?v=4?s=100" width="100px;" alt="Dženan Zukić"/><br /><sub><b>Dženan Zukić</b></sub></a><br /><a href="https://github.com/fepegar/torchio/pulls?q=is%3Apr+reviewed-by%3Adzenanz" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vasl12"><img src="https://avatars.githubusercontent.com/u/15161465?v=4?s=100" width="100px;" alt="vasl12"/><br /><sub><b>vasl12</b></sub></a><br /><a href="#tutorial-vasl12" title="Tutorials">✅</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Avasl12" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://perso.telecom-bretagne.eu/francoisrousseau"><img src="https://avatars.githubusercontent.com/u/398895?v=4?s=100" width="100px;" alt="François Rousseau"/><br /><sub><b>François Rousseau</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Arousseau" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/snavalm"><img src="https://avatars.githubusercontent.com/u/35732360?v=4?s=100" width="100px;" alt="snavalm"/><br /><sub><b>snavalm</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=snavalm" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jcreinhold.com"><img src="https://avatars.githubusercontent.com/u/5241441?v=4?s=100" width="100px;" alt="Jacob Reinhold"/><br /><sub><b>Jacob Reinhold</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=jcreinhold" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Hsuxu"><img src="https://avatars.githubusercontent.com/u/15630477?v=4?s=100" width="100px;" alt="Hsu"/><br /><sub><b>Hsu</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3AHsuxu" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/snipdome"><img src="https://avatars.githubusercontent.com/u/72035308?v=4?s=100" width="100px;" alt="snipdome"/><br /><sub><b>snipdome</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Asnipdome" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/iamSmallY"><img src="https://avatars.githubusercontent.com/u/45689960?v=4?s=100" width="100px;" alt="SmallY"/><br /><sub><b>SmallY</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3AiamSmallY" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/guigautier"><img src="https://avatars.githubusercontent.com/u/45590481?v=4?s=100" width="100px;" alt="guigautier"/><br /><sub><b>guigautier</b></sub></a><br /><a href="#ideas-guigautier" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AyedSamy"><img src="https://avatars.githubusercontent.com/u/55320208?v=4?s=100" width="100px;" alt="AyedSamy"/><br /><sub><b>AyedSamy</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3AAyedSamy" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://delanover.com"><img src="https://avatars.githubusercontent.com/u/3540650?v=4?s=100" width="100px;" alt="J. Miguel Valverde"/><br /><sub><b>J. Miguel Valverde</b></sub></a><br /><a href="#ideas-jmlipman" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/fepegar/torchio/commits?author=jmlipman" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/issues?q=author%3Ajmlipman" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://josegcpa.github.io"><img src="https://avatars.githubusercontent.com/u/40271262?v=4?s=100" width="100px;" alt="José Guilherme Almeida"/><br /><sub><b>José Guilherme Almeida</b></sub></a><br /><a href="#ideas-josegcpa" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/saikhu"><img src="https://avatars.githubusercontent.com/u/24922057?v=4?s=100" width="100px;" alt="Asim Usman"/><br /><sub><b>Asim Usman</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Asaikhu" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cbri92"><img src="https://avatars.githubusercontent.com/u/70302171?v=4?s=100" width="100px;" alt="cbri92"/><br /><sub><b>cbri92</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Acbri92" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/iimog"><img src="https://avatars.githubusercontent.com/u/7403236?v=4?s=100" width="100px;" alt="Markus J. Ankenbrand"/><br /><sub><b>Markus J. Ankenbrand</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Aiimog" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://yanivresearch.info/"><img src="https://avatars.githubusercontent.com/u/338645?v=4?s=100" width="100px;" alt="Ziv Yaniv"/><br /><sub><b>Ziv Yaniv</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=zivy" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LucaLumetti"><img src="https://avatars.githubusercontent.com/u/7543386?v=4?s=100" width="100px;" alt="Luca Lumetti"/><br /><sub><b>Luca Lumetti</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=LucaLumetti" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/commits?author=LucaLumetti" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chagelo"><img src="https://avatars.githubusercontent.com/u/49865033?v=4?s=100" width="100px;" alt="chagelo"/><br /><sub><b>chagelo</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Achagelo" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mueller-franzes"><img src="https://avatars.githubusercontent.com/u/56117447?v=4?s=100" width="100px;" alt="mueller-franzes"/><br /><sub><b>mueller-franzes</b></sub></a><br /><a href="https://github.com/fepegar/torchio/commits?author=mueller-franzes" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wahabk"><img src="https://avatars.githubusercontent.com/u/43846328?v=4?s=100" width="100px;" alt="Abdelwahab Kawafi"/><br /><sub><b>Abdelwahab Kawafi</b></sub></a><br /><a href="https://github.com/fepegar/torchio/issues?q=author%3Awahabk" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the
[all-contributors](https://github.com/all-contributors/all-contributors)
specification. Contributions of any kind welcome!
