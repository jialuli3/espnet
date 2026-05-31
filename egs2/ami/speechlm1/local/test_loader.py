from transformers import AutoConfig, AutoModelForCausalLM

# cache_path = "~/.cache/huggingface/hub"
# LlamaForCausalLM.from_pretrained(
#     "HuggingFaceTB/SmolLM2-1.7B",
#     cache_dir=cache_path,
#     trust_remote_code=True,   # REQUIRED for SmolLM2
# ).get_output_embeddings()

name = "HuggingFaceTB/SmolLM2-1.7B"

cfg = AutoConfig.from_pretrained(name)
#cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
print("model_type:", getattr(cfg, "model_type", None))
print(cfg)

model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
print("Loaded OK:", type(model))
