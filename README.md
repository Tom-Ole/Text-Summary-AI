
based on: https://huggingface.co/docs/transformers/tasks/summarization

Train with:
```bash
CUDA_ALLOC_CONF=expandable_segments:True python3 train.py
```


make sure to have all necessari Python packages installed.
You can find them in the requirement.txt and/or install it with

```bash
pip install -r requirements.txt
```