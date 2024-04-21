# Llama-3-Playground

This will be my playground for all things using LLaMa 3

mamba activate llama3

## Sunday, April 21, 2024

Gonna pull down and play with ...
[cognitivecomputations/dolphin-2.9-llama3-8b](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b)

## Saturday, April 20, 2024

 1) mamba install conda-forge::openai
 2) mamba install conda-forge::pygame
 3) mamba install conda-forge::wandb
 4) mamba install conda-forge::peft
 5) pip install trl
 6) pip install flash-attn

... the currently installed version of bitsandbytes from condaforge was complaining about it not being the gpu version, so I had to remove it, then install the pip version ...

 7) mamba remove bitsandbytes 
 8) pip install bitsandbytes

 Ugh ... killed llama3, then rebuilt ...

 1) mamba create -n llama3 python=3.11
 2) mamba activate llama3
 3) mamba install conda-forge::jupyterlab
 3) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
 4) mamba install conda-forge::datasets
 5) mamba install conda-forge::accelerate
 6) mamba install conda-forge::peft
 7) pip install trl
 8) pip install flash-attn
 9) pip install bitsandbytes
10) mamba install conda-forge::wandb

Now getting an even stranger error in the notebook 'Fine_tune_Llama_3_with_ORPO.ipynb' ... dammit.

OK ... since I struck out with the container 'hfpt_Apr20', gonna go back to the conda container 'llama3' ... and I think the next thing I want to try is ...

[Llama3 in torchtune](https://pytorch.org/torchtune/stable/tutorials/llama3.html)

11) pip install torchtune

I manually copied the hugging face model from ...

    !/Data3/huggingface/transformers/models--meta-llama--Meta-Llama-3-8B$ 

... to 

    ~/Data2/huggingface/hub/models--meta-llama--Meta-Llama-3-8B$ 

Open up a terminal window, activate the llama3 conda environment, and then ran ...

tune download meta-llama/Meta-Llama-3-8B \
    --output-dir /home/rob/Data2/huggingface/hub \
    --hf-token hf_mytokenNot

... hmm nope it wanted to download stuff again .... fack ... and peeking into that target folder, I can see a bunch of new files, so it does seem to be trying to dump stuff into that folder ... 

Hmmm ... ran the following code and got these results ...

        (llama3) rob@KAUWITB:~/Data/Documents/Github/rkaunismaa/Llama-3-Playground$ tune download meta-llama/Meta-Llama-3-8B --output-dir models --hf-token hf_YesThisIsBogus
        /home/rob/miniforge3/envs/llama3/lib/python3.11/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
        warnings.warn(
        Ignoring files matching the following patterns: *.safetensors
        Successfully downloaded model repo and wrote to the following locations:
        /home/rob/Data2/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/561487d18c41c76bcb5fc6cfb73a324982f04f47/original
        (llama3) rob@KAUWITB:~/Data/Documents/Github/rkaunismaa/Llama-3-Playground$ 


Current value of ... 
TRANSFORMERS_CACHE=/home/rob/Data2/huggingface/transformers

 ... also,  [llama-recipies](https://github.com/meta-llama/llama-recipes/tree/main) has been updated for llama3 ...



