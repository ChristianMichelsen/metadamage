# Metadamage - Ancient Damage Estimation for Metagenomics

[![PyPI](https://img.shields.io/pypi/v/metadamage)](https://pypi.org/project/metadamage) ![PyPI - License](https://img.shields.io/pypi/l/metadamage)

---

#### Work in progress. Please contact christianmichelsen@gmail.com for further information.

---


Personal recommendations for this project:

- Python Version: [Pyenv](https://github.com/pyenv/pyenv)
- Virtual Environment:  [Virtualenv](https://github.com/pyenv/pyenv-virtualenv)
- Package Management: [Poetry](https://python-poetry.org/)

This project requires a recent Python version (>=3.8) installed. See more [below](#setup).


## <a name="install"></a> Installation:

With Poetry:
```console
$ poetry add metadamage
```
or, if you prefer regular pip:
```console
$ pip install metadamage
```

## <a name="cli"></a> Metadamage CLI:

For help on the CLI interface, `metadamage` provides a `help` function:

```console
$ metadamage --help
```

Example for fitting a single file using `metadamage`:
```console
$ metadamage fit --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt
```

`metadamage` also allows to fit multiple files:
```console
$ metadamage fit --max-fits 10 --max-cores 2 ./data/input/*.txt
```

## <a name="dash"></a> Dashboard:

To make use of the new, interactive dashboard introduced in version `0.4`, run the following command (after having [fitted](#cli) the files):

```console
$ metadamage dashboard
```

And then open a browser and go to `127.0.0.1:8050` (if it did not open automatically). For more information, use:
```console
$ metadamage dashboard --help
```

## <a name="dash-server"></a> Dashboard on a server:

If you are running metadamage on a server and want to use the dashboard locally, you can setup a SSH tunnel. First, on the server, run `metadamage dashboard` with the relevant options (and keep it open, e.g. with TMUX). Afterwards, on your local machine, run:

```console
$ ssh -L 8050:127.0.0.1:8050 -N user@remote
```
Now you can open a browser and go to [`http://0.0.0.0:8050`](http://0.0.0.0:8050).

In case you're connecting through a jump host, you can use the the `-J` option:
```console
$ ssh -L 8050:127.0.0.1:8050 -N -J user@jumphost user@remote
```

For an easier method, you can setup your ssh config (usually at `~/.ssh/config`) in the following way:

```ssh-config
Host jumphost
    User your-jumphost-username-here
    HostName your-jumphost-address-here

Host remote

Host dashboard
    Port 22
    LocalForward 8050 localhost:8050
    RemoteCommand echo "Connecting to dashboard ... CTRL+C to terminate"; sleep infinity

Host remote dashboard
    ProxyJump jumphost
    User your-remote-username-here
    HostName your-remote-address-here
```

Now if you simply run the following on your own computer (in a new terminal session):

```console
$ ssh dashboard
```
you can open open a browser and go to [`http://0.0.0.0:8050`](http://0.0.0.0:8050).

<!-- ssh -L 8050:127.0.0.1:8050 -N -J willerslev mnv794@wonton-snm -->
<!-- ssh -L 8050:127.0.0.1:8050 -N -J mnv794@ssh-snm-willerslev.science.ku.dk mnv794@wonton-snm -->

<!-- ssh -L 8050:127.0.0.1:8050 -N hep -->



## <a name="options"></a> Metadamage CLI fit options:

The `metadamage fit` CLI has the following options.

- Output directory
  - `--out_dir`: The directory in which the fit results are stored. Default location is `./data/out`. Do not change unless you known what you are doing.

- Maximum values
  - `--max-fits`: Maximum number of fits to do. Default is None, i.e. fit everything.
  - `--max-cores`: Maximum number of cores to use while fitting. Default is 1.
  - `--max-position`: Maximum position in the sequence to include. Default is +/- 15 (forward/reverse).

- Minimum values or cuts/thresholds for plots
  - `--min-alignments`: Minimum number of alignments (![N_\mathrm{alignments}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+N_%5Cmathrm%7Balignments%7D)
) of a single TaxID to be fitted. Default is 10.
  - `--min-k-sum`: Minimum sum of `k` of a single TaxID to be fitted. Here `k` might be e.g. the number of A→T transitions in the forward direction and the number of G→A transitions in the reverse direction.  In that case, it would be: ![\mathtt{k} = \sum_{z=1}^{15} \left( N_{\mathrm{CT}}(z)  +  N_{\mathrm{GA}}(-z) \right)](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cmathtt%7By%7D+%3D+%5Csum_%7Bz%3D1%7D%5E%7B15%7D+%5Cleft%28+N_%7B%5Cmathrm%7BCT%7D%7D%28z%29++%2B++N_%7B%5Cmathrm%7BGA%7D%7D%28-z%29+%5Cright%29).  Default is 10.
- `--min-N-at-each-pos`: Minimum number of N at each position.

- Other:
  - `--substitution-bases-forward`:  Which substitution to check for damage in the forward region. Do not change this value except for control checks. Default is `CT`.
  - `--substitution-bases-reverse`:  Which substitution to check for damage in the reverse region. Do not change this value except for control checks. Default is `GA`.

- Boolean Flags
  - `--forced`: Force redo everything (count data and fits).
  - `--dask-port`: Local port used for Dask. Integer. Default is 8787.
  <!-- - `--force-reload-files`: Force reload the original datafile (and thus not load the autogenerated parquet file). -->
  <!-- - `--force-plots`: Force recreate  the plots, even though the plots already exist. -->
  <!-- - `--force-no-plots`: Force not to make any plots at all, even though `max-plots` exists (and is >0). -->

  - `--version`: Print the current version of the program and exit.


<!-- [tex-image-link-generator](https://tex-image-link-generator.herokuapp.com/) -->
<!-- https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b -->

---

## <a name="setup"></a> Setup Local Python Environment:

Make sure you have a local Python environment. Personally, I recommend using Pyenv for installing Python versions and Pyenv-Virtualenv for easy managing of virtuel environments. See e.g. [this](https://github.com/pyenv/pyenv-installer#pyenv-installer) for easy installation of both.

Make sure you have a decent Python version (>=3.8) installed:
```console
$ pyenv install 3.8.7
```

Now we set up a virtual environment, such that changes you do in this environment does not affect your other Python projects:
```console
$ pyenv virtualenv 3.8.7 metadamage38
$ pyenv activate metadamage38
```

We now use Poetry to setup a new project which uses metadamage. Follow the interactive guide:
```console
$ poetry new metadamage-folder
$ cd metadamage-folder
```

Instead of activating the environment manually after every new login, we can tell `pyenv` to remember it for us:
```console
$ pyenv local metadamage38
```

We now have a working local, virtual Python environment where the packages are managed by Poetry, so we can now add metadamamage to our project:
```console
$ poetry add metadamage
```

At this point you should log out of your terminal and log in again for reloading everything. Now if you just write:

```console
$ metadamage
```

you should see the following:

```console
$ metadamage
Usage: metadamage [OPTIONS] FILENAMES...
Try 'metadamage --help' for help.

Error: Missing argument 'FILENAMES...'.
```

which shows that it is working and installed. You can now use `metadamage --help` for more help (together with the variable explanations [above](#options)).

---

## <a name="update"></a> Update:

With Poetry:
```console
$ poetry update metadamage
```
or, if you prefer regular pip:
```console
$ pip install metadamage --upgrade
```

---

## <a name="dev_branch"></a> Development Branch:

You can also use a newer version directly from Github:
```console
$ poetry add git+https://github.com/ChristianMichelsen/metadamage.git
```
or a specific branch (named `BRANCH`):
```console
$ poetry add git+https://github.com/ChristianMichelsen/metadamage.git#BRANCH
```

---

## Conda:

If you prefer using Conda, you can also install `metadamage` (via pip). First create a folder:
```console
$ mkdir metadamage-conda
$ cd metadamage-conda
```

To install `metadamage`:
```console
$ wget https://raw.githubusercontent.com/ChristianMichelsen/metadamage/main/environment.yaml
$ conda env create -f environment.yaml
```

To update it to a new, released version of `metadamage`:
```console
$ wget https://raw.githubusercontent.com/ChristianMichelsen/metadamage/main/environment.yaml
$ conda env update --file environment.yaml
```

Finally remember to activate the environment:
```console
$ conda activate metadamage
```

<!-- poetry add git+https://github.com/christianmichelsen/metadamage.git -->
<!-- poetry add git+https://github.com/christianmichelsen/metadamage.git#develop -->
