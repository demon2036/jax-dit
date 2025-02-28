#echo "${SCRIPT_PATHS[@]}"
#SCRIPT_PATHS=$1
SCRIPT_PATHS=("$@")
#echo "Array content: $1"

#echo "Array content: ${SCRIPT_PATHS[@]}"
for script in "${SCRIPT_PATHS[@]}"; do
    echo " $script"
    pkill -9 -f python
    sudo rm /tmp/libtpu_lockfile
    source ~/miniconda3/bin/activate base;
#    python -u test.py

#    python -u test_orbax.py
    python -u tsne_fork_pjit.py


#    python -u main.py --yaml-path $script
#    python -u main_test.py --yaml-path $script

done
