# GNAMES (Genetic-Nurture and Assortative-Mating-Effects Simulator) `beta v0.1`

`gnames` is a Python 3.x package for efficient simulation of multigenerational data under assortative mating and genetic nurture.

## Installation

:warning: Before downloading `gnames`, please make sure [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/) with **Python 3.x** are installed. For now, `gnames` only requires the packages `numpy`, `pandas` and `time`.

In order to download `gnames`, open a command-line interface by starting [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/), navigate to your working directory, and clone the `gnames` repository using the following command:

```  
git clone https://github.com/devlaming/gnames.git
```

Now, enter the newly created `gnames` directory using:

```
cd gnames
```
You can now run the following commands, to test if `gnames` is functioning properly:

```
python -c "from gnames import gnames; gnames.Test()"
```

This command should yield output along the following lines:
```
Test of gnames with 1000 founders and 10000 SNPs
For 2 offspring generations
With 2 children per mating pair
Initialising simulator
Drawing allele frequencies SNPs founders
Drawing true SNP effects
Drawing genotypes founders
-> block 1 out of 1
Drawing traits generation 0
Performing assortative mating generation 0
-> for group 1 out of 1
Highest diagonal element of GRM for founders = 1.042
Simulating data for 2 subsequent generations
Drawing genotypes children for generation 1
-> for set of children 1 out of 2
-> for set of children 2 out of 2
Drawing traits generation 1
Performing assortative mating generation 1
-> for group 1 out of 2
-> for group 2 out of 2
Drawing genotypes children for generation 2
-> for set of children 1 out of 2
-> for set of children 2 out of 2
Drawing traits generation 2
Performing assortative mating generation 2
-> for group 1 out of 2
-> for group 2 out of 2
Highest diagonal element of GRM after 2 generations = 1.043
Runtime: 1.365 seconds
```

This output shows `gnames` has simulated a founder population comprising 1000 individuals and 10,000 SNPs. Subsequently, `gnames` has simulated two generations of offspring data under genetic nurture and assortative mating. The highest element of the diagonal of the GRM has increased from 1.042 to 1.043 over the two generations.

## Tutorial

Once `gnames` is up-and-running, you can simply incorporate it in your Python code, as illustrated in the following bit of Python code:

```
from gnames import gnames
import numpy as np
import matplotlib.pyplot as plt

N=1000
M=10000
T=1000

gsimulator=gnames(N,M)
vDiags0=np.sort(gsimulator.ComputeDiagsGRM())

gsimulator.Simulate(T)
vDiags1000=np.sort(gsimulator.ComputeDiagsGRM())

plt.plot(np.vstack((vDiags0,vDiags1000)).T)
```

Please observe that this code may take between five and twenty minutes to run, as it simulates data on 10,000 SNPs for 1000 founders, after which 1000 (!) subsequent generations of offspring data are drawn.

The plot at the end of the code shows the diagonal elements of the GRM sorted from small to large for the founders (blue line) and for the 1000th offspring generation (orange line).

## Updating `gnames`

You can update to the newest version of `gnames` using `git`. First, navigate to your `gnames` directory (e.g. `cd gnames`), then run
```
git pull
```
If `gnames` is up to date, you will see 
```
Already up to date.
```
otherwise, you will see `git` output similar to 
```
remote: Enumerating objects: 8, done.
remote: Counting objects: 100% (8/8), done.
remote: Compressing objects: 100% (4/4), done.
remote: Total 6 (delta 2), reused 6 (delta 2), pack-reused 0
Unpacking objects: 100% (6/6), 2.82 KiB | 240.00 KiB/s, done.
From https://github.com/devlaming/gnames
   481a4bf..fddd8cc  main       -> origin/main
Updating 481a4bf..fddd8cc
Fast-forward
 README.md | 128 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 gnames.py |  26 ++++++++++++-
 2 files changed, 153 insertions(+), 1 deletion(-)
 create mode 100644 README.md
```
which tells you which files were changed.

If you have modified the `gnames` source code yourself, `git pull` may fail with an error such as `error: Your local changes [...] would be overwritten by merge`. 

## Support

Before contacting me, please try the following:

1. Go over the tutorial in this `README.md` file
2. Go over the method, described in *tba* (citation below)

### Contact

In case you have a question that is not resolved by going over the preceding two steps, or in case you have encountered a bug, please send an e-mail to r\[dot\]devlaming\[at\]vu\[dot\]nl.

## Citation

If you use the software, please cite the manuscript in which `gnames` is described:

*tba*

## License

This project is licensed under GNU GPL v3.

## Authors

Ronald de Vlaming (Vrije Universiteit Amsterdam)
