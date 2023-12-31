# LangchainDocs

## HuggingFace Pipeline - Alternative to TextGenerationWebUI

```
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#HuggingFace Model ID
model_id = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# torch.bfloat16 - Ampere+ GPU
# torch.float16 - 8bit or older GPU

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir='./models', #If using local folder for loading input it here
    torch_dtype=torch.float16, # torch.bfloat16 - Ampere+ GPU // torch.float16 - 8bit or older GPU
    trust_remote_code=True, #Requirement for some models
    load_in_8bit=True, #Loads in 8bit mode instead of fp16 // Ex: 7b Model = 13gb in fp16, 6-7gb in 8bit 
    device_map="auto", #adapts memory for multiple GPU's
    offload_folder="offload" #Offload to CPU and RAM if needed ***Really Slow
    )

# Set model to inference mode "Evaluation"
model.eval()

# HuggingFace Pipeline 
personalpipe = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
```

## Langchain Conversation Chain (Subset of Memory)
https://python.langchain.com/docs/modules/memory/

```
# Conversation chain expects "input" and "history" as written as is to be able to store in the BufferMemory
template = PromptTemplate(template="<|prompter|>{input}<|endoftext|><|assistant|>{history}", input_variables=["input","history"])

# k=2 means that it stores the last two recieved prompts
nexuschatchain = ConversationChain(
    llm = llms,
    prompt=template,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)
```
