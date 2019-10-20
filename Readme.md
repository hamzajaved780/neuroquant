<p align="center"><img src="http://i.imgur.com/fcot8Jw.png" width="800" align="middle"></p>

Implementation of a topological quantum processor on [Intel Loihi chip](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423&tag=1), a dedicated neuromorphic hardware for training and inference of Spiking Neural Networks.


Built using : https://www.nengo.ai/nengo-loihi/

### How to run?
>~~~~
>pip install -r requirements.txt
>~~~~
To execute code on the remote Loihi Superhost please configure your machine using the following [instruction](https://www.nengo.ai/nengo-loihi/installation.html). After a succesfull installation run the code on Superhost by adding SLURM=1 to the command:

`SLURM=1 python Neuromorphic_TQP.py`
