# Wrapper

ModelWrapper bridges the engine step_node to the model layer, loading base model/tokenizer/hypernet from config and exposing adapter generation, LoRA hot-swap, and (continuation) generation with all GPU imports deferred.

::: rune.model.wrapper
