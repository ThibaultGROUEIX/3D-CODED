🚀 Last upgrade 🚀 : Release of data + dataloaders
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/3d-coded-3d-correspondences-by-deep-1/3d-point-cloud-matching-on-faust)](https://paperswithcode.com/sota/3d-point-cloud-matching-on-faust?p=3d-coded-3d-correspondences-by-deep-1)

# 3D-CODED : 3D Correspondences by Deep Deformation :page_with_curl:

This repository contains the source codes for the paper [3D-CODED : 3D Correspondences by Deep Deformation](http://imagine.enpc.fr/~groueixt/3D-CODED/index.html). The task is to put 2 meshes in point-wise correspondence. Below, given 2 humans scans with holes, the reconstruction are in correspondence (suggested by color).

<img src="README/mesh25.ply.gif" style="zoom:60%" /><img src="README/25RecBestRotReg.ply.gif" style="zoom:60%" />

<img src="README/mesh8.ply.gif" style="zoom:60%" /><img src="README/8RecBestRotReg.ply.gif" style="zoom:60%" />

# Easy Install :construction_worker:

```shell
git clone git@github.com:ThibaultGROUEIX/3D-CODED.git ## Download the repo
conda create --name pytorch-atlasnet python=3.7 ## Create python env
source activate pytorch-atlasnet
pip install pandas visdom trimesh sklearn shapely
conda install pytorch torchvision -c pytorch # or from sources if you prefer
# you're done ! Congrats :)
```

Tested on 11/18 with  pytorch 0.4.1 (py37_py36_py35_py27__9.0.176_7.1.2_2) and [latest source](

#### Build chamfer distance 

```shell
source activate pytorch-atlasnet
cd 3D-CODED/extension
python setup.py install
```

# Using the Trained models :train2:

The trained models and some corresponding results are also available online :

- [The trained_models](https://cloud.enpc.fr/s/n4L7jqD486V8IJn) go in ``` trained_models/```

#### On the demo meshes 

Require 3 GB of RAM on the GPU and 17 sec to run (Titan X Pascal). 

```shell
cd trained_models; ./download_models.sh; cd .. # download the trained models
cd data; ./download_template.sh; cd .. # download the template
python inference/correspondences.py
```
This script takes as input 2 meshes from ```data``` and compute correspondences in ```results```. Reconstruction are saved in ```data```

It should look like :

* **Initial guesses** for *example0* and *example1*:

<img src="README/example_0InitialGuess.ply.gif" style="zoom:80%" /><img src="README/example_1InitialGuess.ply.gif" style="zoom:80%" />

* **Final reconstruction** for *example0* and *example1*:

<img src="README/example_0FinalReconstruction.ply.gif" style="zoom:80%" /><img src="README/example_1FinalReconstruction.ply.gif" style="zoom:80%" />


#### On your own meshes

You need to make sure your meshes are preprocessed correctly :

* The meshes are loaded with **Trimesh**, which should support a bunch of formats, but I only tested ```.ply``` files. Good converters include [**Assimp**](https://github.com/assimp/assimp) and [Pymesh](https://github.com/qnzhou/PyMesh).


* The trunk axis is the **Y axis** (visualize your mesh against the mesh in ```data``` to make sure they are normalized in the same way). 
* the **scale** should be about 1.7 for a standing human (meaning the unit for the point cloud is the ```cm```). You can automatically scale them with the flag ```--scale 1```


#### Options  

```python
'--HR', type=int, default=1, help='Use high Resolution template for better precision in the nearest neighbor step ?'
'--nepoch', type=int, default=3000, help='number of epochs to train for during the regression step'
'--model', type=str, default = 'trained_models/sup_human_network_last.pth',  help='your path to the trained model'
'--inputA', type=str, default =  "data/example_0.ply",  help='your path to mesh 0'
'--inputB', type=str, default =  "data/example_1.ply",  help='your path to mesh 1'
'--num_points', type=int, default = 6890,  help='number of points fed to poitnet'
'--num_angles', type=int, default = 100,  help='number of angle in the search of optimal reconstruction. Set to 1, if you mesh are already facing the cannonical 				direction as in data/example_1.ply'
'--env', type=str, default="CODED", help='visdom environment'
'--clean', type=int, default=0, help='if 1, remove points that dont belong to any edges'
'--scale', type=int, default=0, help='if 1, scale input mesh to have same volume as the template'
'--project_on_target', type=int, default=0, help='if 1, projects predicted correspondences point on target mesh'

```



#### Failure modes instruction : :warning:

- Sometimes the reconstruction is flipped, which break the correspondences. In the easiest case where you meshes are registered in the same orientation, you can just fix this angle in ```reconstruct.py``` line 86, to avoid the flipping problem. Also note from this line that the angle search only looks in [-90°,+90°].

- Check the presence of lonely outliers that break the Pointnet encoder. You could try to remove them with the ```--clean``` flag.


#### Last comments

* If you want to use ```inference/correspondences.py``` to process a hole dataset, like FAUST test set, make sure you don't load the same network in memory every time you compute correspondences between two meshes (which will happen with the naive and simplest way of doing it by calling ```inference/correspondences.py``` iteratively on all the pairs). A example of bad practice is in ```./auxiliary/script.sh```, for the FAUST inter challenge. **Good luck :-)**



# Train the models 

#### Data: the easy way

```shell
cd data; ./download_data.sh; cd .. # download the data
```

### Data: the hard way

The hard way is generating your own synthetic dataset of humans.

Since the generation process of the dataset is quite heavy, it has it's own README in ```data/README.md```. Brace yourselve :-)

#### Start training

* First launch a visdom server :

```bash
python -m visdom.server -p 8888
```

* Launch the training. Check out all the options in ```./training/train_sup.py``` .

```shell
source activate pytorch-atlasnet; env=3D-CODED
python ./training/train_sup.py --env $env  |& tee ${env}.txt
```

* Monitor your training on http://localhost:8888/

![visdom](./README/1532524819586.png)

* Timings, results, memory requirements

| Method         | Faust euclidean error in cm | GPU memory | Time by epoch⁽²⁾ |
| -------------- | --------------------------- | ---------- | ---------------- |
| train_sup.py   | 2.878                       | TODO       | TODO             |
| train_unsup.py | 4.883                       | TODO       | TODO             |

⁽²⁾this is only an estimate, the code is not optimised

#### Options

```python
'--batchSize', type=int, default=32, help='input batch size'
'--workers', type=int, help='number of data loading workers', default=8
'--nepoch', type=int, default=75, help='number of epochs to train for'
'--model', type=str, default='', help='optional reload model path'
'--env', type=str, default="unsup-symcorrect-ratio", help='visdom environment'
'--laplace', type=int, default=0, help='regularize towords 0 curvature, or template curvature'
```



## Acknowledgement

* The code for the Chamfer Loss was adapted from **[Fei Xia](http://fxia.me/)**'a repo : [PointGan](https://github.com/fxia22/pointGAN). Many thanks to him !
* The code for the Laplacian regularization comes from [**Angjoo** **Kanazawa**](https://people.eecs.berkeley.edu/~kanazawa/) and [**Shubham** **Tulsiani**](https://people.eecs.berkeley.edu/~shubhtuls/). This was so helpful, thanks !
* Part of the SMPL parameters used in the training data comes from [**Gül** **Varol**](https://www.di.ens.fr/~varol/)'s repo : https://github.com/gulvarol/surreal But most of all, thanks for all the advices :)
* The FAUST Team for their prompt reaction in resolving a benchmark issue the week of the deadline, especially to [**Federica** **Bogo**](https://ps.is.tuebingen.mpg.de/person/fbogo) and **Jonathan Williams**.
* The efficient code for to compute geodesic errors comes from  https://github.com/zorah/KernelMatching. Thanks!
* The [SMAL](http://smalr.is.tue.mpg.de/) team, and [SCAPE](https://ai.stanford.edu/~drago/Projects/scape/scape.html) team for their help in generating the training data.
* [DeepFunctional Maps](https://arxiv.org/abs/1704.08686) authors for their fast reply the week of the rebuttal ! Many thanks.
* **[Hiroharu Kato](http://hiroharu-kato.com/projects_en/neural_renderer.html)** for his very clean neural renderer code, that I used for the gifs :-)
* [Pytorch developpers](https://github.com/pytorch/pytorch) for making DL code so easy.
* This work was funded by [Ecole Doctorale MSTIC](http://www.univ-paris-est.fr/fr/-ecole-doctorale-mathematiques-et-stic-mstic-ed-532/). Thanks !
* And last but not least, my great co-authors :  **[Matthew Fisher](http://graphics.stanford.edu/~mdfisher/publications.html), [Vladimir G. Kim](http://vovakim.com/), [Bryan C. Russell](http://bryanrussell.org/), and [Mathieu Aubry](http://imagine.enpc.fr/~aubrym/cv.html)**



## Citing this work 

If you find this work useful in your research, please consider citing:

```
@inproceedings{groueix2018b,
          title = {3D-CODED : 3D Correspondences by Deep Deformation},
          author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G. and Russell, Bryan and Aubry, Mathieu},
          booktitle = {ECCV},
          year = 2018}
        }
```

# 

## Cool Contributions

* **[Zhongshi Jiang](https://cs.nyu.edu/~zhongshi/)** applying trained model on a monster model :japanese_ogre: (**left**: original , **right:** reconstruction)

![visdom](./README/image.png)

[![Analytics](https://ga-beacon.appspot.com/UA-91308638-2/github.com/ThibaultGROUEIX/3D-CODED/readme.md?pixel)](https://github.com/ThibaultGROUEIX/3D-CODED/)

## License

[MIT](https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/license_MIT)

## 