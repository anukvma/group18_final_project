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

## Training Details

### Bart
**Code File**: [Group18FineTuneBartEmailSubjectFinal.ipynb](https://github.com/anukvma/group18_email_subject_generation/blob/main/Group18FineTuneBartEmailSubjectFinal.ipynb) \
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
Test:
```
{'rouge1': 38.3593,
 'rouge2': 17.502,
 'rougeL': 31.3383,
 'rougeLsum': 33.4465,
 'gen_len': 40.5906}
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

## Observations
1. Generative models are very large to be trained on base model, we had to use quantized versions. Also for training we used [PEFT](https://huggingface.co/docs/peft/en/package_reference/lora) 

     
## HuggingFace Demo URL
**Gradio App:** https://huggingface.co/spaces/GSridhar1982/EmailSubjectGenerationDemo \
Code Files: \
[GradioAppWithModelSelection.ipynb](https://github.com/anukvma/group18_email_subject_generation/blob/main/GradioAppWithModelSelection.ipynb): Gradio App Notebook with model selection option. \
[Group18EmailSubjectGradioApp.ipynb](https://github.com/anukvma/group18_email_subject_generation/blob/main/Group18EmailSubjectGradioApp.ipynb): Gradio App without model selection \

**FAST API:** https://anukvma-emailsubjectapi.hf.space \
Code files: [api(folder)](https://github.com/anukvma/group18_email_subject_generation/tree/main/api): Code for Fast API includes Dockerfile, requirements.txt and main.py \
Curl command for API call:
```
curl --location --request GET 'https://anukvma-emailsubjectapi.hf.space' \
--header 'Content-Type: application/json' \
--data-raw '{
    "model_name":"anukvma/bart-base-medium-email-subject-generation-v5",
    "email_content": "Harry - I got kicked out of the system, so I'\''m sending this from Tom'\''s account. He can fill you in on the potential deal with STEAG. I left my resume on your chair. I'\''ll e-mail a copy when I have my home account running. My contact info is:"
}'
```
