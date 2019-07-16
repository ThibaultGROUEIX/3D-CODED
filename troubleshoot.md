# 1. running ```python test_chamfer.py```, you get a ```Segmentation fault (core dumped)```

Double-check two things:

* your GCC and G++ version are up to date (version 7 at least).

* your pytorch package cuda version matches your ```/usr/local/cuda``` version

  You can check the former and the latter with:

  ```python
  import torch
  torch.version.cuda
  ```

  

  ```
cat /usr/local/cuda/version.txt
  ```

 #2. Compiling chamfer -> ```cc1plus: error: /usr/lib/x86_64-linux-gnu/libcudnn.so.5/include: Not a directory```

From @palanglois. https://github.com/ThibaultGROUEIX/3D-CODED/issues/4

It's very likely that this error happens because you either set the environment variable 'CUDNN_HOME' or 'CUDNN_PATH'.

From there, 2 possibilites.

- You can remove the env variables, but it is likely that you use it for other projects.
- Otherwise, you can try to add the following lines at the beginning of your setup.py in order to delete the env variables only in the child process :

```
import os`
`if 'CUDNN_HOME' in os.environ: del os.environ['CUDNN_HOME']`
`if 'CUDNN_PATH' in os.environ: del os.environ['CUDNN_PATH']
```

#3. Importing chamfer ->  

```
>>> import chamfer`
`Traceback (most recent call last): File "<stdin>", line 1, in <module> ImportError: /home/adrian/research/envs/research/lib/python3.7/site-packages/chamfer-0.0.0-py3.7-linux-x86_64.egg/chamfer.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN2at19UndefinedTensorImpl10_singletonE
```

* try to `import torch` first