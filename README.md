# LangchainDocs

##HuggingFace Pipeline - Alternative to TextGenerationWebUI

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"

tokenizer = AutoTokenizer.from_pretrained(model_id)

#bfloat16 - Ampere+ GPU
#float16 - 8bit or older GPU

model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./models', 
    torch_dtype=torch.float16, trust_remote_code=True, load_in_8bit=True, device_map="auto", offload_folder="offload")

# Set PT model to inference mode
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
