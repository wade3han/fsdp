apt-get update
apt-get install git
pip install torch transformers
git clone https://github.com/wade3han/fsdp
cd fsdp
torchrun --nproc_per_node=4 optimizer_save.py; torchrun --nproc_per_node=4 optimizer_load.py
