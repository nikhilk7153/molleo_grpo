# MolLEO

[arXiv] [Efficient Evolutionary Search Over Chemical Space with Large Language Models](https://arxiv.org/abs/2406.16976)

[Website] [MolLEO Project](https://molleo.github.io/)

## About

MolLEO is an LLM-augmented evlotuionary algorithm for molecular discovery!

![image](images/README/molleo_overview.gif)

## Setups
You need to get an OpenAI API key for GPT-4. BioT5 is an open-source language model which can work on either GPU or CPU. Qwen is supported via vLLM server for efficient local inference. Currently, this code repo does not support MolLEO(MOLSTM), but we will update soon.

### Package Installation
```bash
conda create -n molleo python=3.9
conda activate molleo
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install PyTDC 
pip install PyYAML
pip install rdkit
pip install transformers
pip install sentencepiece
pip install selfies
pip install vllm
pip install requests
```

Then we can activate conda via following command. 
```bash
conda activate molleo 
```

### Starting vLLM Server for Qwen (Required for Qwen usage)
Before using Qwen, you need to start the vLLM server in a separate terminal:

**Option 1: Use the startup script (recommended)**
```bash
conda activate molleo
./start_qwen_server.sh
```

**Option 2: Start manually**
```bash
conda activate molleo
python single_objective/vllm_server.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

Keep this server running in a separate terminal. The server will be accessible at `http://localhost:8000`.


### Experiments
The experiments are conducted on the following categories: `single obejective optimization` and `multi objective optimization`.

To run experiments on single objective optimization task:

```bash
cd single_objective
# BioT5 on jnk3 task
python run.py molleo --mol_lm BioT5 --oracles jnk3 --seed 1 2 3
# GPT-4 on gsk3b task
python run.py molleo --mol_lm GPT-4 --oracles gsk3b --seed 1 2 3
# Qwen on jnk3 task
python run.py molleo --mol_lm Qwen --oracles jnk3 --seed 1 2 3
```
To run experiments on multi objective optimization task:

```bash
cd multi_objective
# objective summation on task 1
python run.py molleo_multi --mol_lm BioT5 --min_obj sa --max_obj jnk3 qed --seed 1 2 3
# pareto optimal set selection on task 2
python run.py molleo_multi_pareto --mol_lm GPT-4 --min_obj sa --max_obj gsk3b qed --seed 1 2 3
# pareto optimal set selection on task 3
python run.py molleo_multi_pareto --mol_lm GPT-4 --min_obj sa gsk3b drd2 --max_obj jnk3 qed --seed 1 2 3
# objective summation with Qwen
python run.py molleo_multi --mol_lm Qwen --min_obj sa --max_obj jnk3 qed --seed 1 2 3
```

## Citation
If you find our work helpful, please consider citing our paper:

```
@misc{wang2024efficientevolutionarysearchchemical,
      title={Efficient Evolutionary Search Over Chemical Space with Large Language Models}, 
      author={Haorui Wang and Marta Skreta and Cher-Tian Ser and Wenhao Gao and Lingkai Kong and Felix Streith-Kalthoff and Chenru Duan and Yuchen Zhuang and Yue Yu and Yanqiao Zhu and Yuanqi Du and Alán Aspuru-Guzik and Kirill Neklyudov and Chao Zhang},
      year={2024},
      eprint={2406.16976},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
      url={https://arxiv.org/abs/2406.16976}, 
}
```
