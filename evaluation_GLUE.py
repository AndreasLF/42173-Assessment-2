import sys
from transformers import AlbertForSequenceClassification, AlbertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate  # Use evaluate instead of load_metric
import torch
import pdb

# Check if a task name was provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python evaluate_glue.py <glue_task>")
    sys.exit(1)

# Get the GLUE task name from the command-line argument
glue_task = sys.argv[1]

# Map the number of labels for specific tasks
num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

if glue_task not in num_labels_map:
    print(f"GLUE task '{glue_task}' is not supported in this script.")
    sys.exit(1)

# Load the pre-trained ALBERT model and tokenizer
model_name = "albert-base-v2"
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels_map[glue_task])
tokenizer = AlbertTokenizer.from_pretrained(model_name)

# Load the GLUE dataset
dataset = load_dataset("glue", glue_task)

# Load the evaluation metric
metric = evaluate.load("glue", glue_task)

# Tokenization for sentence-pair tasks vs single sentence tasks
def preprocess_function(examples):
    if glue_task == "mnli":
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding=True, max_length=512)
    elif glue_task == "qnli":
        return tokenizer(examples['question'], examples['sentence'], truncation=True, padding=True, max_length=512)
    elif glue_task == "qqp":  
        return tokenizer(examples['question1'], examples['question2'], truncation=True, padding=True, max_length=512)
    elif glue_task in ["mrpc", "rte", "wnli"]:
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True, max_length=512)
    else:  # Single sentence tasks like CoLA, SST-2
        return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=512)


# Tokenize the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Determine the correct evaluation dataset split based on the task
if glue_task == "mnli":
    eval_dataset = encoded_dataset["validation_matched"]  # MNLI has both 'matched' and 'mismatched' validation sets
else:
    eval_dataset = encoded_dataset["validation"]  # For most other tasks like QNLI, QQP, RTE, etc.


# Define evaluation arguments
training_args = TrainingArguments(
    output_dir=f"./results_{glue_task}",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
    logging_dir=f"./logs_{glue_task}",
)

def compute_metrics(p):
    predictions = torch.tensor(p.predictions)  # Convert to a torch.Tensor
    predictions = torch.argmax(predictions, axis=1)
    
    # Compute Matthews correlation coefficient (already done by Hugging Face evaluate)
    mcc_result = metric.compute(predictions=predictions, references=p.label_ids)
    
    # Compute accuracy
    accuracy = (predictions == torch.tensor(p.label_ids)).float().mean().item()
    
    # Add accuracy to results
    mcc_result["accuracy"] = accuracy
    
    return mcc_result


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Evaluate the pre-trained model without fine-tuning
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Evaluation Results on {glue_task.upper()} Task: {eval_results}")
