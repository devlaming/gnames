# GNAMES (Genetic-Nurture and Assortative-Mating-Effects Simulator) `beta v0.1`

`gnames` is a Python 3.x package for efficient simulation of multigenerational data under assortative mating and genetic nurture effects.

## Installation

:warning: Before downloading `gnames`, please make sure [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/) with **Python 3.x** are installed.

In order to download `gnames`, open a command-line interface by starting [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/), navigate to your working directory, and clone the `gnames` repository using the following command:

```  
git clone https://github.com/devlaming/gnames.git
```

Now, enter the newly created `gnames` directory using:

```
cd gnames
```

Then run the following commands to create a custom Python environment which has all of `gnames`'s dependencies (i.e. an environment that has packages `numpy`, `pandas`, `scipy`, and `tqdm` pre-installed):

```
conda env create --file gnames.yml
conda activate gnames
```

(or `activate gnames` instead of `conda activate gnames` on some machines).

In case you cannot create a customised conda environment (e.g. because of insufficient user rights) or simply prefer to use Anaconda Navigator or `pip` to install packages e.g. in your base environment rather than a custom environment, please note that `gnames` only requires Python 3.x with the packages `numpy`, `pandas`, `scipy`, and `tqdm` installed.

You can now run the following commands, to test if `gnames` is functioning properly:
```
python -c "from gnames import gnames; gnames.Test()"
```

This command should yield output along the following lines:
```
TEST OF GNAMES
with 1000 founders, 10,000 SNPs, and two children per pair
INITIALISING SIMULATOR
Drawing alleles for SNPs of founders
Drawing allele frequencies for SNPs of founders
Drawing true SNP effects
Drawing genotypes founders
Highest diagonal element of GRM for founders = 1.056
SIMULATING 10 GENERATIONS
100%|████████████████████████████████████| 10/10 [00:04<00:00,  2.46it/s]
Highest diagonal element of GRM after 10 generations = 1.09
GENERATING OUTPUT
Calculating and storing classical GWAS and within-family GWAS
results based on offspring data last generation
Writing PLINK files (genotypes.bed,.bim,.fam,.phe)
Making GRM in GCTA binary format (genotypes.grm.bin,.grm.N.bin,.grm.id)
Runtime: 4.7 seconds
```

This output shows `gnames` simulated a founder population comprising 1000 individuals and 10,000 SNPs. Subsequently, `gnames` simulated ten generations of offspring data under genetic nurture and assortative mating. `gnames` reports that the highest element of the diagonal of the GRM increased from 1.056 to 1.09 over the ten generations.

In addition, `gnames` performed a classical GWAS and a within-family GWAS based on the offspring data for the last generation. Results are exported to human-readable files: `results.GWAS.classical.txt` and `results.GWAS.within_family.txt`.

Moreover, `gnames` created a set of PLINK binary files: `genotypes.bed`, `genotypes.bim`, `genotypes.fam`. These PLINK binary files can readily be used for follow-up analyses using tools such as [PLINK](https://www.cog-genomics.org/plink/). `gnames` also created a phenotype file, `genotypes.phe`, that can be used by PLINK e.g. to perform a GWAS.

Finally, `gnames` created a set of GRM files in GCTA binary format: `genotypes.grm.id`, `genotypes.grm.bin`, and `genotypes.grm.N.bin`. These files combined with `genotypes.phe` can easily be used for follow-up analyses using tools such as [MGREML](https://www.github.com/devlaming/mgreml) and [GCTA](https://yanglab.westlake.edu.cn/software/gcta/).

The whole simulation, two GWASs, and export to PLINK binary files and to GCTA binary GRM files took less than five seconds.

## Tutorial

Once `gnames` is up-and-running, you can simply incorporate the tool in your Python code, as illustrated in the following bit of Python code:

```
from gnames import gnames
import numpy as np
import matplotlib.pyplot as plt

N=1000
M=10000
T=1000
F=0.4

gsimulator=gnames(N,M,dMAF0=F)
vDiags0=np.sort(gsimulator.ComputeDiagsGRM())

gsimulator.Simulate(T)
vDiags1000=np.sort(gsimulator.ComputeDiagsGRM())

plt.plot(np.vstack((vDiags0,vDiags1000)).T)
plt.savefig('diagsGRM.pdf')
plt.close()

gsimulator.PerformGWAS('n1000.m10000.t1000')
gsimulator.MakeBed('n1000.m10000.t1000')
gsimulator.MakeGRM('n1000.m10000.t1000')
```

:warning: This code may take about ten minutes to run, as the code simulates data on 10,000 SNPs for 1000 founders, after which 1000 (!) subsequent generations of offspring data are drawn.

The plot that is created near the end of the code shows the diagonal elements of the GRM sorted from small to large for the founders (blue line) and for the 1000th offspring generation (orange line). As a result of strong assortative mating in this simulation, we can see that the diagonal elements of the GRM have considerably shifted away from one over the generations.

In addition, this bit of code shows how `gnames` can be used to calculate GWAS summary statistics based on the last generation, here yielding files named `n1000.m10000.t1000.GWAS.classical.txt` and `n1000.m10000.t1000.GWAS.within_family.txt`.

Finally, the code also shows how `gnames` can be used to create PLINK binary files and binary GRM files for the last generation. These files are here named (*i*) `n1000.m10000.t1000.bed`, `n1000.m10000.t1000.bim`, and `n1000.m10000.t1000.fam` and (*ii*) `n1000.m10000.t1000.grm.id`, `n1000.m10000.t1000.grm.bin`, and `n1000.m10000.t1000.grm.N.bin`.

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
