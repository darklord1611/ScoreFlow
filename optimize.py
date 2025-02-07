import torch
import re
import argparse
import os
import pickle
import random
import yaml
from datasets import Dataset
from ScoreFlow.DPOtrainer import DPOTrainer
from ScoreFlow.DPOconfig import DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_checkpoint_in_epoch_dir(epoch_dir, str_id):
    subdirs = [os.path.join(epoch_dir, d) for d in os.listdir(epoch_dir)
               if os.path.isdir(os.path.join(epoch_dir, d)) and d.startswith(str_id)]
    if len(subdirs) == 1:
        return subdirs[0]
    elif len(subdirs) == 0:
        raise FileNotFoundError(f"No checkpoint directory found in '{epoch_dir}' starting with {str_id}.")
    else:
        raise ValueError(f"Multiple checkpoint directories found in '{epoch_dir}', expected only one: {subdirs}")

def training(epoch, config1):

    next_epoch = str(int(epoch) + 1)

    # load model from huggingface
    model = AutoModelForCausalLM.from_pretrained(
        config1["generator"]["model"],
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(config1["generator"]["model"])

    ensure_directory_exists("scoreflow_workspace/finetuned/" + next_epoch)

    # get LoRA
    if epoch == "0":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.01,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(model, "scoreflow_workspace/finetuned/" + epoch + "/checkpoint")
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        model.train()

    # get preference data
    with open("scoreflow_workspace/output_preference_data/preference_data-" + epoch + ".pkl", 'rb') as file:
        data = pickle.load(file)
    random.shuffle(data)
    dataset = Dataset.from_list(data)

    dpo_args = DPOConfig(output_dir="scoreflow_workspace/finetuned/" + next_epoch,
                              logging_steps=10,
                              save_steps=140000,
                              per_device_train_batch_size = 1,
                              per_device_eval_batch_size = 1,
                              num_train_epochs = 1,
                              use_score = True
                        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    trainer = DPOTrainer(model=model,
                         args=dpo_args, 
                         processing_class=tokenizer, 
                         train_dataset=dataset
                        )
    trainer.train()

def merge(epoch, config1):
    next_epoch = str(int(epoch) + 1)
    base_model_name = config1["generator"]["model"]
    target_dir = "scoreflow_workspace/finetuned/" + next_epoch
    peft_model_path = find_checkpoint_in_epoch_dir(target_dir, "checkpoint-")
    output_model_path = target_dir + "/merged"
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Merge
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    os.rename(peft_model_path, "scoreflow_workspace/finetuned/" + next_epoch + "/checkpoint")

def main():
    # input epoch(0, 1, 2...)
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("--epoch", type=str, required=True, help="Value for Epoch")
    args = parser.parse_args()
    epoch = args.epoch

    # load model configurations
    with open("config/config1.yaml", "r") as file:
        config1 = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = config1["CUDA_VISIBLE_DEVICES"]
    
    training(epoch, config1)

    # merge LoRA to base model
    merge(epoch, config1)

if __name__ == "__main__":
    main()
