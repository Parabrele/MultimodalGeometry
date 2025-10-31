# test if the prompt var is not set
export PATH=/home/thibaut.boissin/overleaves/texlive/2023/bin/x86_64-linux:$PATH

if [ -z "$PS1" ]; then
    # prompt var is not set, so this is *not* an interactive shell
    return
fi

parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "

source /etc/profile.d/cuda.sh
set_cuda_version 12.1 8.9.2

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

source_env() {
    : #do things with parameters like $1 such as
# source $1/bin/activate  # commented out by conda initialize
}
export PATH=/home/thibaut.boissin/.local/bin:$PATH

# uncomment this to activate the virtual env
# ./home/thibaut.boissin/envs/myenv_1/bin/activate

alias list_cuda_versions='ls /usr/local/cudnn'
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

launch_jupyter() {
    jupyter notebook --no-browser --port=8888
}
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniconda/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

