# LangchainDocs

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
