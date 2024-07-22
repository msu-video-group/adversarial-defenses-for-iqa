## A branch for running defenses
### Structure
* `/methods` - this folder is not used for launches
* `/subjects` - the folder with target quality metrics
* `/defences` - the folder with defenses. Contains defense subfolders similar to attacks. Also inside there is a CI file, a utils folder with scripts used for the run
    * `/*название защиты*/` -- the name of a defense should not contain capital letters and underscores, use `-.` to separate the words 
        * `run.py` - there must be an `run.py` file in the defense folder, containing the `Defense` class with the implemented `__call__` method, as well as the code calling the `test_main` function (see any protection)
        * `dfsrc/` - if the defense implementation requires any additional scripts or files, they must be located in this folder. It is copied to the docker image without changes, and it will be available during initialization and use of defense. If you do not need any additional files for defense, you do not need to create this folder. If you need to load weights for defense, add the script `setup.sh` to the `dfsrc`, pumping their wget from server. The script will not run by itself - add its launch when initializing defense or at the beginning of the file `run.py `using `subprocess.run()` (see mprnet or fcn defense)
    * `ci.yml` - a CI file similar to the attacks. If you are adding a new defense, make an appropriate entry in it. Note that if there is a `-` in the protection name, then in the line `-if: $*protection name* == "yes"` in the defense name `-` are replaced by `_` 
    * `utils/` -- the folder with the main scripts for launching defenses
        * `defence_presets.json` - defense presets
        * `defence_evaluate.py` - the main script, an analog of `fgsm_evaluate.py` for attacks
        * `defence_dataset.py` - the class of the dataset used when running defenses on the attacked dataset
        * `defence_scoring_methods.py` - functions for calculating different scores, by which we compare defenses
        * `Dockerfile` - the main docker file that defines the image for the defense run
        * `read_dataset.py`, `evaluate.py`, `metrics.py` - other auxiliary files
* `/scripts` - basic bash scripts and lists of defenses/metrics/attacks. From the important:
    * `attack-test.sh` - скрипт, запускающий test джобы защит. В начале этого скрипта находятся все основные параметры, контролирующие прогон: пресет, батч сайз, пути до датасетов и тд.
    * `defences.txt` - список защит. При добавлении новой защиты нужно добавить сюда название.
