# LangchainDocs

## Langchain Conversation Chain (Memory Subset)
https://python.langchain.com/docs/modules/memory/

```
#Conversation chain expects "input" and "history" as written to be able to store in the BufferMemory
template = PromptTemplate(template="<|prompter|>{input}<|endoftext|><|assistant|>{history}", input_variables=["input","history"])

nexuschatchain = ConversationChain(
    llm = llms,
    prompt=template,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)
```
