from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import argparse

def merge_and_save(base_model_name_or_path, adapter_path, save_path, use_flash_attn, tokenizer_revision, trust_remote_code, use_slow_tokenizer):
    print("Loading base model...")
    base_condig = AutoConfig.from_pretrained(base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        config=base_condig,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )
    print("Loading adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        is_trainable=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        revision=tokenizer_revision,
        trust_remote_code=trust_remote_code,
        use_fast=not use_slow_tokenizer,
    )
    
    print("Merging model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Merge a LoRA adapter into a base model')
    
    parser.add_argument('--base-model-name-or-path', 
                        type=str, 
                        required=True,
                        help='Path to the base model')
    
    parser.add_argument('--adapter-path', 
                        type=str, 
                        required=True,
                        help='Path to the LoRA adapter')
    
    parser.add_argument('--save-path', 
                        type=str, 
                        required=True,
                        help='Path to save the merged model')
    
    parser.add_argument('--use_flash_attn',
                        default=True,
                        help='Whether to use flash attention in the model training')
    
    args = parser.parse_args()

    merge_and_save(
        base_model_name_or_path=args.base_model_name_or_path,
        adapter_path=args.adapter_path,
        save_path=args.save_path,
        use_flash_attn=args.use_flash_attn,
        tokenizer_revision="main",
        trust_remote_code=False,
        use_slow_tokenizer=True
    )

if __name__ == '__main__':
    main()