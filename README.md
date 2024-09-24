# ALBERT GLUE Task Evaluation Script

This script evaluates the pre-trained ALBERT model on various **GLUE** benchmark tasks. It is part of the **Advanced NLP** course at **UTS**. Due to limited resources, no fine-tuning has been performed; this script demonstrates that fine-tuning is essential for achieving high accuracy on these tasks.

## Requirements

1. Install required libraries:
    ```bash
    pip install transformers datasets evaluate torch
    ```

2. Download the pre-trained ALBERT model and the GLUE datasets directly from Hugging Face.

## Usage

Run the script by passing the desired GLUE task as an argument:

```bash
python evaluate_glue.py <glue_task>
