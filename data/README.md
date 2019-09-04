# Data Generation for Training of 3D-CODED



### The Case of Human : SMPL and FAUST



* **FAUST** Data can be downloaded [here](http://faust.is.tue.mpg.de/challenge/Inter-subject_challenge/datasets), after having created an account.

* **SMPL** generation code can be downloaded [here](http://smpl.is.tue.mpg.de/).

  ```shell
  conda create --name smpl python=2.7 ## SMPL use python 2
  conda install opencv
  pip install chumpy
  #then download smpl
  ```
  In your python code, to use smpl, add it to your path :
  ```python
  import sys
  sys.path.append("PATH_TO_SMPL")
  ```
  You can also find in the repo the trained smpl models :


  ```shell
	basicModel_f_lbs_10_207_0_v1.0.0.pkl
	basicmodel_m_lbs_10_207_0_v1.0.0.pkl
  ```

Place these two files under `data/smpl_data` folder.

* **SURREAL** pose and shape parameters can be downloaded from Gül Varol's [repo](https://github.com/gulvarol/surreal), in which we're interested in ```smpl_data.npz```. 

  Place the file under `data/smpl_data` folder.

you should now have :

```bash
smpl_data/
├── basicModel_f_lbs_10_207_0_v1.0.0.pkl
├── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
└── smpl_data.npz 
```

Once you have everything set up and installed ( Chumpy, SMPL, trained models, parameters from surreal), you should be able to call

```shell
python generate_data_humans.py
```

It generates 230 000 training humans (and 400 validation humans) :

- 100 000 male humans using poses and shapes from SURREAL using ``` generate_surreal```
- 100 000 female humans using poses and shapes from SURREAL ``` generate_surreal```
- 15 000 bent male humans ``` generate_benthuman```
- 15 000 bent female humans ``` generate_benthuman```



The code also includes generation methods using random gaussians for the pose parameters, instead of parameters from SURREAL. I tried it quickly and got less good results. The variety of shapes is much better than in SURREAL but we often get unrealistic poses. 



### The Case of Animals : SMAL

* Download the models from the [website](http://smal.is.tue.mpg.de/), after having created an account. You're especially looking for ``` smal_CVPR2017.pkl ``` , the trained model.
* You can now launch : 

```shell
python generate_data_animals.py
```

It generates 200 000 training hyppos (and 200 validation hyppos). Of course, you can play around with other categories !



This whole thing is a bit tricky to set up, so i advise you to dive in the code and ask questions :-)