To not blow up disk space I redefine some envs to point to a larger partition:

```
export CACHE_DIR=~/projects/cache
export HF_DATASETS_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
export DIFFUSERS_CACHE=$CACHE_DIR
```
