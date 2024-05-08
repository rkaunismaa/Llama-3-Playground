# Llama-3-Playground

This will be my playground for all things using LLaMa 3

mamba activate llama3

## Wednesday, May 8, 2024

Looking at [Meta Llama Recipies](https://github.com/meta-llama/llama-recipes) for something to run. I will place these examples into the 'meta-llama/llama-recipies' folder.

19) pip install groq

I then took the 'Prompt_Engineering_with_Llama_3.ipynb' notebook that works nicely with groq and copied it to 'Prompt_Engineering_with_Llama_3_LMStudio.ipynb' then tweaked that to work with LMStudio and not groq. This works just fine. 

20) pip install openai


## Tuesday, April 30, 2024

Working through the notebook from the Sam Witteveen video [Llama 3 - 8B & 70B Deep Dive](https://www.youtube.com/watch?v=8Ul_0jddTU4).

* YT_Llama_3_8B_Testing.ipynb

## Monday, April 29, 2024

Working through the notebook referenced from the youtube video [Fine Tune Llama 3 using ORPO](https://www.youtube.com/watch?v=nPIGVaYPQAg)

Also, remember, [Meta Llama Recipies](https://github.com/meta-llama/llama-recipes) contains other useful notebook examples and scripts, now updated for Llama 3.

## Wednesday, April 24, 2024

Working through Rag-Chatbot.ipynb.

 12) mamba install conda-forge::gradio
 13) mamba install conda-forge::sentence-transformers
 14) pip install spaces
 15) mamba install pytorch::faiss-gpu

 The install of faiss-gpu changed a lot of libraries in this environment .... 

        Package         Version  Build                         Channel           Size
        ─────────────────────────────────────────────────────────────────────────────────
        Install:
        ─────────────────────────────────────────────────────────────────────────────────

        + libfaiss        1.8.0  h5aaf3ed_0_cuda11.4.4         pytorch          346MB
        + faiss-gpu       1.8.0  py3.11_hedc54c9_0_cuda11.4.4  pytorch            5MB

        Change:
        ─────────────────────────────────────────────────────────────────────────────────

        - libblas         3.9.0  16_linux64_mkl                conda-forge     Cached
        + libblas         3.9.0  20_linux64_mkl                conda-forge       15kB
        - liblapack       3.9.0  16_linux64_mkl                conda-forge     Cached
        + liblapack       3.9.0  20_linux64_mkl                conda-forge       14kB
        - libcblas        3.9.0  16_linux64_mkl                conda-forge     Cached
        + libcblas        3.9.0  20_linux64_mkl                conda-forge       14kB
        - liblapacke      3.9.0  16_linux64_mkl                conda-forge     Cached
        + liblapacke      3.9.0  20_linux64_mkl                conda-forge       14kB
        - blas-devel      3.9.0  16_linux64_mkl                conda-forge     Cached
        + blas-devel      3.9.0  20_linux64_mkl                conda-forge       14kB

        Upgrade:
        ─────────────────────────────────────────────────────────────────────────────────

        - llvm-openmp    15.0.7  h0cdce71_0                    conda-forge     Cached
        + llvm-openmp    18.1.3  h4dfa4b3_0                    conda-forge       58MB
        - mkl-include  2022.1.0  h84fe81f_915                  conda-forge     Cached
        + mkl-include  2023.2.0  h84fe81f_50496                conda-forge      705kB
        - mkl          2022.1.0  h84fe81f_915                  conda-forge     Cached
        + mkl          2023.2.0  h84fe81f_50496                conda-forge      164MB
        - mkl-devel    2022.1.0  ha770c72_916                  conda-forge     Cached
        + mkl-devel    2023.2.0  ha770c72_50496                conda-forge       30kB
        - blas            2.116  mkl                           conda-forge     Cached
        + blas            2.120  mkl                           conda-forge       14kB

        Downgrade:
        ─────────────────────────────────────────────────────────────────────────────────

        - pytorch         2.2.2  py3.11_cuda11.8_cudnn8.7.0_0  pytorch         Cached
        + pytorch         2.0.1  py3.11_cuda11.8_cudnn8.7.0_0  pytorch            2GB
        - torchtriton     2.2.0  py311                         pytorch         Cached
        + torchtriton     2.0.0  py311                         pytorch           66MB
        - torchaudio      2.2.2  py311_cu118                   pytorch         Cached
        + torchaudio      2.0.2  py311_cu118                   pytorch            8MB
        - torchvision    0.17.2  py311_cu118                   pytorch         Cached
        + torchvision    0.15.2  py311_cu118                   pytorch            9MB

Dammit .... now the model does not load ... I think I am gonna rebuild this environment!

Yup. Instruct.ipynb which did work is now failing with the same error message. Fack! Gonna kill llama3 and rebuild! ... 


 1) mamb env remove -n llama3
 2) mamba create -n llama3 python=3.11
 3) mamba activate llama3
 4) mamba install conda-forge::jupyterlab
 5) mamba install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
 6) mamba install conda-forge::faiss-gpu
 7) mamba install conda-forge::transformers
 8) mamba install conda-forge::sentence-transformers

  ... up to this point, everything installed cleanly without any problems ...

 9) mamba install conda-forge::accelerate
10) mamba install conda-forge::bitsandbytes
11) mamba install conda-forge::wandb
12) mamba install conda-forge::peft
13) pip install spaces ... (hmm this installed gradio, matplotlib, pydantic, and a ton of other needed libraries!)

... hmm now getting that error about bitsandbytes ...

    Could not find the bitsandbytes CUDA binary at PosixPath('/home/rob/miniforge3/envs/llama3/lib/python3.11/site-packages/bitsandbytes/libbitsandbytes_cuda118.so')
    The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.

14) mamba remove bitsandbytes ... this ran without any issues 
15) pip install bitsandbytes

... OK Nice! That fixed the problem! ...

16) mamba install conda-forge::ipywidgets
17) pip install trl
18) pip install flash-attn


Whelp, rebooted, and ran 'pip install -qqq flash-attn' on the llama3 environment and now the notebook 'Fine_tune_Llama_3_with_ORPO.ipynb' is running! Nice! ... I really wanted to see if I could get this to run locally, and it does! Training puts a real load on the 4090! 

And 'Fine_tune_Llama_3_with_ORPO.ipynb' all runs! Great! (I aborted the upload of the model to HuggingFace cuz it would take forever!)

## Sunday, April 21, 2024

Gonna pull down and play with ...
[cognitivecomputations/dolphin-2.9-llama3-8b](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b)

12) mamba install conda-forge::ipywidgets

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



