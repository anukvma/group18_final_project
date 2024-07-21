# Email Subject Generation (Group 18)

## Description
Generate a succinct subject line from the body of an email.
Email Subject Line Generation task involves identifying the most important sentences in an email and abstracting their message into just a few words. The project provides an opportunity to work with generative models in NLP, specifically using GPT-2 variants, and to explore different metrics for evaluating text generation.

## DataSet
Dataset used is from the below repository for fine tuning the models
The Annotated Enron Subject Line Corpus: https://github.com/ryanzhumich/AESLC

## Models
The following models are fine tuned 
| LLM     	| Framework             | Model Type        | Training Steps       	| Evaluation Method    	| 
|---------	|---------------------	|-------------------|---------------------	|----------------------	|
| Mistral 	| unsloth             	| 4 bit quantized 	| 60 	                  | ROUGE Score         	|
| Llama3  	| unsloth             	| 4 bit quantized 	| 60  	                | ROUGE Score           |
| T5      	| HuggingFace           | Base model       	| 200                  	| ROUGE Score          	|
| Bart    	| HuggingFace           | Base model       	| 200                  	| ROUGE Score          	|

## Result
| LLM     	| Rogue1              	| Rogue2               	| RougeL              	| RogueLSum            	|
|---------	|---------------------	|----------------------	|---------------------	|----------------------	|
| Mistral 	| 0.04175057546404236 	| 0.015307029349338995 	| 0.03865576026979294 	| 0.040112317820734385 	|
| Llama3  	| 0.044540652323630435 	| 0.016282087086038018 	| 0.03984053234184394  	| 0.04157418257161926  	|
| T5      	| 0.144567            	| 0.070306             	| 0.140258            	| 0.141119             	|
| Bart    	| 0.267373            	| 0.134597             	| 0.249993            	| 0.250012             	|
