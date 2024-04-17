# Knowledge Graph Construction by Triple and Event Extraction using Few-Shot & CoT Techniques


# Table of Contents

- [Team](#team)
- [Acknowledgement](#acknowledgement)
- [3rd Party Libraries](#3rd-party-libraries)
- [Requirements](#requirements)
- [Execution](#execution)
  - [Event Extraction](#event-extraction)
  - [RE_TACRED Triple Extraction](#re_tacred-triple-extraction)
  - [SciERC Triple Extraction](#scierc-triple-extraction)
- [Data](#data)
  - [Event Extraction](#event-extraction-1)
  - [Triple Extraction](#triple-extraction)
    - [SciERC Dataset](#scierc-dataset)
    - [RE_TACRED Dataset](#re_tacred-dataset)


## Team
| Student name              |     CCID     |
|---------------------------|--------------|
| Jai Riley                 |    jrbuhr    |
| MohammadJavad Ardestani   |   Ardestan   |



## Acknowledgement

We acknowledge the following external resources consulted for this project:
- Baseline prompts and initial data preprocessing were sourced from the AUTOKG repository.
- All LLMs were utilized exclusively via API calls, eliminating the need for local execution. To generate API keys for the LLMs used in this project, refer to the following links:
    - Gemini API (`gemini-pro`) <[Link](https://ai.google.dev/tutorials/get_started_web)>
    - OpenAI API (`gpt-3.5-turbo-0125` & `gpt-4`) <[Link](https://www.datacamp.com/tutorial/guide-to-openai-api-on-tutorial-best-practices)>
    - LLaMA API - Cloudflaure (`llama-2-13b-chat-awq`) <[Link](https://developers.cloudflare.com/workers-ai/get-started/)>
- ChatGPT was employed to aid in code documentation, crafting docstrings, and reviewing the report.

## 3rd Party Libraries

We utilized the following 3rd-party libraries for this project:
- `scikit-learn` library for computing Precision, Recall, and F1 score.
- `langchain_google_genai` library for prompting `gemini-pro`.
- `OpenAI` library for prompting `gpt-3.5-turbo-0125` & `gpt-4`.
- `requests` library for sending HTTP requests to Cloudflare API to prompt `llama-2-13b-chat-awq`.

## Requirements
Run the following command to install all the required packages.

`pip install -r requirements.txt`



## Execution

List of prompt_types:
- `zero_shot`
- `few_shot`
- `CoT`

List of LLMs:
- `Gemini`
- `LLaMA`
- `GPT`

If you want to prompt GPT's LLMs, you need to mention which GPT model you want as the last input argument. For other LLMs, you do not need to mention any sub-model.

List of GPT's sub-models:
- `gpt-4`
- `gpt-3.5-turbo-0125`

All of the below commands are case-sensitive, so be careful to pass input arguments exactly as mentioned in the above lists.

### Event Extraction

Use the following command in the current directory to run the code for **Event Extraction** part:
`python3 src/event_extraction.py <prompt_type> <LLM> <LLM's API> <GPT's sub-models>`

Example usage: For running a zero-shot prompt on `gpt-4': 

`python3 src/event_extraction.py zero_shot GPT <LLM's API> gpt-4`

Example usage: For running a few-shot prompt on `Gemini': 

`python3 src/event_extraction.py few_shot Gemini <LLM's API>`

### RE_TACRED Triple Extraction

Use the following command in the current directory to run the code for **Triple Extraction** From `RE_TACRED` dataset:
```python3 src/triple_extraction_retacered.py <prompt_type> <LLM> <LLM's API> <GPT's sub-models>```


### SciERC Triple Extraction

Use the following command in the current directory to run the code for **Triple Extraction** From `SciERC` dataset:
`python3 src/triple_extraction_scierc.py <prompt_type> <LLM> <LLM's API> <GPT's sub-models>`

## Output Structure
If you run the commands above for each dataset, you'll find a main directory for that dataset alongside the 'src' folder. Inside each dataset directory, the outputs are located within specific folders named after the language models (LLMs) that were used.
If you execute the commands above for each dataset and each LLM, you'll achieve the following file structure:

```
MAVAN_Experiments_Log/
    Gemini/
    gpt-3.5-turbo-0125/
    gpt-4/
    LLaMA/
RE_TACRED_Experiments_Log/
    Gemini/
    gpt-3.5-turbo-0125/
    gpt-4/
    LLaMA/
SciERC_Experiments_Log/
    Gemini/
    gpt-3.5-turbo-0125/
    gpt-4/
    LLaMA/
src/
```
## Data

The sample set for each dataset is embedded in its related .py file. Below, you can access the entire dataset.

### Event Extraction
**MAVEN Dataset**
- Link to dataset: https://github.com/zjunlp/AutoKG/tree/main/KG%20Construction/MAVEN/datas

- Sample: 
```
{
    "sentence": "French troops were sent into the area, as Général d'armée Maxime Weygand attempted to build up a defence in depth on the south bank of the Somme and make bigger attacks to eliminate the German bridgeheads.",
    "events": [
        {
            "Event_type": "Sending",
            "trigger_word": "sent"
        },
        {
            "Event_type": "Building",
            "trigger_word": "build"
        },
        {
            "Event_type": "Creating",
            "trigger_word": "make"
        },
        {
            "Event_type": "Destroying",
            "trigger_word": "eliminate"
        }
    ],
    "id": 0
}

```
### Triple Extraction 
**SciERC Dataset**
- Link to dataset: https://github.com/zjunlp/AutoKG/tree/main/KG%20Construction/SciERC/datas
- Sample:
```
{
    "id": 2,
    "relation": "USED-FOR",
    "tokens": "This paper describes a method for incorporating priming into an incremental probabilistic parser .",
    "h": {
        "name": "priming",
        "pos": [7, 8]
    },
    "t": {
        "name": "incremental probabilistic parser",
        "pos": [10, 13]
    }
}

```
<br><br>
**RE_TACRED Dataset**
- Link to dataset: https://github.com/zjunlp/AutoKG/tree/main/KG%20Construction/RE-TACRED/datas

- Sample: 
```
{
    "id": 416,
    "relation": "org:city_of_branch",
    "tokens": "Even at its peak , in 2007 , ALICO 's portfolio of credit-default swaps was just a fraction of the one at AIG Financial Products , the London shop whose collapsing business led the U.S. government to prop up AIG , the biggest bailout in American history .",
    "h": {
        "name": "ALICO",
        "pos": [8, 9]
    },
    "subj_type": "ORGANIZATION",
    "t": {
        "name": "London",
        "pos": [27, 28]
    },
    "obj_type": "CITY"
}

```
