# AIML Questions and Answers (Group 18)

## Description
Answer the given AIML Questions: Modeling a domain-specific GPT-variant model that can answer the questions specific to the AIML course. It has been observed that while pretrained models can produce relevant textual output for general, open-domain textual prompts, the models lack the capability of producing finer outputs when it comes to domain-specific tasks. For this purpose, we commonly finetune the model on a dataset specific to that task, to tailor its expertise on it. Here, the participants will work together to build a novel, relevant dataset for the task. Post finetuning, they will observe its performance on unseen, related questions. \
**Project Proposal**: [Group 18 - Capstone Project proposal - Google Docs.pdf](https://github.com/anukvma/group18_email_subject_generation/blob/main/Group%2018%20-%20Capstone%20Project%20proposal%20-%20Google%20Docs.pdf)

## DataSet
Dataset used is from the below repository for fine tuning the models
[https://drive.google.com/drive/folders/1O2qMvEfKXyhdF1HcHIck6eWLPRMGmPUv?usp=sharing](https://drive.google.com/drive/folders/1O2qMvEfKXyhdF1HcHIck6eWLPRMGmPUv?usp=sharing)

## Models
The following models are fine tuned 
| LLM     	| Framework             | Model Type        | Training Steps       	| Evaluation Method    	| 
|---------	|---------------------	|-------------------|---------------------	|----------------------	|
| Bart    	| Transformer           | Base model       	| 600                  	| ROUGE Score          	|
| GPT2    	| Transformer (LoRa)    | Base model       	| 600                  	| ROUGE Score          	|
| Mistral7b | Unsloth               | 4bit Quantized   	| 60                 	| ROUGE Score          	|
| Llama3.1	| Unsloth               | 4bit Quantized   	| 60                  	| ROUGE Score          	|


## Training Details

### Bart
**Code File**: [Group18FineTuneBartQuestionAnswer.ipynb](https://github.com/anukvma/group18_final_project/blob/main/aiml_question_answers/Group18FineTuneBartQuestionAnswer.ipynb) \
Model: facebook/bart-large-xsum \
Training Framework: Transformer Seq2SeqTrainer \
Training Arguments: 
```
Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=6,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)
```
**Result**:
Training:
```
Step	Training Loss	Validation Loss	Rouge1	Rouge2	Rougel	Rougelsum	Gen Len
100	3.014600	2.570830	33.137500	14.415600	27.374400	28.190300	31.850000
200	2.087700	2.501255	35.015200	15.145300	28.741700	29.822700	34.110000
300	1.507400	2.540349	35.207000	14.807500	28.700600	30.715900	41.570000
400	1.089400	2.717226	34.006500	14.434000	27.868000	29.710000	41.610000
500	0.776600	2.945216	36.228900	16.209900	29.720400	31.375800	40.040000
600	0.577400	3.062018	35.786700	16.206200	29.868600	31.719700	41.520000
```
### GPT
**Code File**: [Group18FineTuneGPTQuestionAnswer.ipynb](https://github.com/anukvma/group18_final_project/blob/main/aiml_question_answers/Group18FineTuneGPTQuestionAnswer.ipynb) \
Model: gpt2 \
Training Framework: Transformer \
Training Arguments: 
```
lora_config = LoraConfig(
    r=4,  # Rank of the low-rank adaptation matrix
    lora_alpha=16,  # Scaling factor for the low-rank adaptation
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",  # No bias adjustment
    task_type="CAUSAL_LM"  # Task type for GPT-like models
)
TrainingArguments(
    output_dir="./gpt3-lora-qa",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    num_train_epochs=5,
    per_device_train_batch_size=2,  # Lower batch size
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Adjust batch size based on GPU memory
    save_steps=500,
    save_total_limit=2,
    fp16=True,  # Use mixed precision training for efficiency
    report_to="none",
    dataloader_pin_memory=True
)
```
**Result**:
Training:
```
Step	Training Loss	Validation Loss	Rouge1	Rouge2	Rougel	Rougelsum
100	1.561300	1.344851	0.476642	0.168196	0.388193	0.421086
200	1.515700	1.325351	0.477955	0.170701	0.389714	0.421446
300	1.494900	1.314155	0.482422	0.173687	0.392187	0.425295
400	1.484500	1.308874	0.482999	0.176904	0.394567	0.426415
500	1.476100	1.306868	0.482330	0.177874	0.394122	0.427295
```

### Llama3.1
**Code File**: [Group18_AIML_Q_&_A_Llama_3_1_8b_finetuning.ipynb](https://github.com/anukvma/group18_final_project/blob/main/aiml_question_answers/Group18_AIML_Q_&_A_Llama_3_1_8b_finetuning.ipynb) \
Model: Llama3.1 \
Training Framework: Unsloth \
Training Arguments: 
```
SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    )
```
**Result**:
Training
```
Step	Training Loss	Validation Loss	Rouge1	Rouge2	Rougel	Rougelsum
60	    0.88	        1.55	        0.29    0.10    0.22    0.23
```
### Mistral7b
**Code File**: [Group18_AIML_Q_&_A__Mistral_7b_finetuning.ipynb](https://github.com/anukvma/group18_final_project/blob/main/aiml_question_answers/Group18_AIML_Q_&_A_Mistral_7b_finetuning.ipynb) \
Model: Mistral7b \
Training Framework: Unsloth \
Training Arguments: 
```
SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    )
```
**Result**:
Training
```
Step	Training Loss	Validation Loss	Rouge1	Rouge2	Rougel	Rougelsum
60	    0.88	        1.55	        0.26    0.08    0.21    0.20
```
## Inference Results

### Question:
﻿What is a linear classifier? 

### Bart Model Output:
A linear classifier is a model used to classify data points along a line.


## Model Evaluation Criteria
### Rouge Score
Rouge score measures the similarity between the generated subject and the provided subject using overlapping n-grams. It ranges from 0 to 1, with higher values indicating better summary quality.
## Result
| LLM     	| Rogue1              	| Rogue2               	| RougeL              	| RogueLSum            	|
|---------	|---------------------	|----------------------	|---------------------	|----------------------	|
| Bart    	| 0.383593            	| 0.17502            	| 0.313383           	| 0.405906             	|
| GPT2    	| 0.482330            	| 0.177874           	| 0.394122          	| 0.427295             	|
| Llama31   | 0.20            	    | 0.10           	    | 0.22        	        | 0.23             	    |
| Mistral7b | 0.26            	    | 0.08           	    | 0.21       	        | 0.20             	    |

## Observations
1. Generative models are very large to be trained on base model, we had to use quantized versions. Also for training we used [PEFT](https://huggingface.co/docs/peft/en/package_reference/lora) 
2. We observed GPT2 is best model for questions and answers task followed by bart.
     
## HuggingFace Demo URL
**Gradio App:** [anukvma/Question_Answers](https://huggingface.co/spaces/anukvma/Question_Answers) \

**FAST API:** [hugging face space](https://huggingface.co/spaces/anukvma/AIMLQnAAPI) \
Code files: [api(folder)](https://github.com/anukvma/group18_final_project/tree/main/aiml_question_answers/api): Code for Fast API includes Dockerfile, requirements.txt and main.py \
Curl command for API call:
```
curl --location --request GET 'https://anukvma-aimlqnaapi.hf.space' \
--header 'Content-Type: application/json' \
--data-raw '{
    "question": "what is linear regression?"
}'
```
Response: "Linear regression is a statistical method used to forecast the probability of a dependent variable using a linear equation."
