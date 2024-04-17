from sklearn.metrics import  precision_score, recall_score, f1_score
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from openai import OpenAI
import copy
import requests
import sys


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/375121881c6ec963b21944159717e15e/ai/run/"

zero_shot_prompt = '''
The list of event types: {}
Given a sentence what types of events are included in this sentence? Please return the most likely answer according to the list of event types above. Require the answer in the form: Event type:
Given a sentence: {}
Event type:
'''

few_shot_prompt = '''
The list of event types: {}
What types of events are included in the following sentence? Please return the most likely answer according to the list of event types above. Require the answer in the form: Event type.Ensure that the extracted event is included in the prepared list.
Example:

Given sentence: "French troops were sent into the area, as Général d'armée Maxime Weygand attempted to build up a defence in depth on the south bank of the Somme and make bigger attacks to eliminate the German bridgeheads."
Event type: Sending, Building, Creating, Destroying

Given sentence: "The Fall of Philadelphia marked the fall of Philadelphia, the last independent Christian Greek settlement in western Asia Minor, to the Muslim Turks of the Ottoman Sultanate."
Event type: Recording, Motion_directional

Given sentence: "The report cites research that shows people's feelings about a police encounter depend significantly on whether they feel the officer displays respect and courtesy."
Event type: Change_sentiment, Adducing, Influence


Given sentence: {}
Event type:
'''

CoT_prompt = '''

Analyze the provided sentence step by step to extract the event types:

Step 1: Recognize Trigger Words: Trigger words are terms or phrases indicating the start or occurrence of events within a text or context.
Step 2: Match each trigger word with the most relevant event type from the list of events. The list of event types: {}
output: Format the output by listing the event types separated by commas after the keyword "Event type: "

Example 1 :
given sentence: "French troops were sent into the area, as Général d'armée Maxime Weygand attempted to build up a defence in depth on the south bank of the Somme and make bigger attacks to eliminate the German bridgeheads."

Step1: triger words: <sent>, <build>, <make>, <eliminate>

Step2:
For the trigger word <sent>, the most related event among events in the Event list is <Sending>.
For the trigger word <build>, the most related event among events in the Event list is <Building>.
For the trigger word <make>, the most related event among events in the Event list is <Creating>.
For the trigger word <eliminate>, the most related event among events in the Event list is <Destroying>.

Event type: Sending, Building, Creating, Destroying


Example 2:
given sentence: "The report cites research that shows people's feelings about a police encounter depend significantly on whether they feel the officer displays respect and courtesy."

step1:  triger words: <cites>, <depend>, <feel>

Step2:
For the trigger word <cites>, the most related event among events in the Event list is <Adducing>.
For the trigger word <depend>, the most related event among events in the Event list is <Influence>.
For the trigger word <feel>, the most related event among events in the Event list is <Change_sentiment>.

Event type: Adducing, Influence, Change_sentiment

given sentence: {}
'''

MAVEN_dataset_40samples = [
{'sentence': "French troops were sent into the area, as Général d'armée Maxime Weygand attempted to build up a defence in depth on the south bank of the Somme and make bigger attacks to eliminate the German bridgeheads.", 'events': [{'Event_type': 'Sending', 'trigger_word': 'sent'}, {'Event_type': 'Building', 'trigger_word': 'build'}, {'Event_type': 'Creating', 'trigger_word': 'make'}, {'Event_type': 'Destroying', 'trigger_word': 'eliminate'}], 'id': 0},
{'sentence': 'From the end of June to early September, over 3,000 forest fires were recorded across the nation.', 'events': [{'Event_type': 'Recording', 'trigger_word': 'recorded'}], 'id': 1},
{'sentence': 'The invasion of Kuwait on 2 August 1990 was a two-day operation conducted by Iraq against the neighboring State of Kuwait, which resulted in the seven-month-long Iraqi occupation of the country.', 'events': [{'Event_type': 'Action', 'trigger_word': 'conducted'}, {'Event_type': 'Causation', 'trigger_word': 'resulted in'}], 'id': 2},
{'sentence': 'Market Garden consisted of two sub operations: The attack was the largest airborne operation up to that point in World War II.', 'events': [{'Event_type': 'Attack', 'trigger_word': 'attack'}], 'id': 3},
{'sentence': 'The hurricane continued to strengthen while developing a well-defined eye, and peaked as a Category 4 hurricane on September 11.', 'events': [{'Event_type': 'Cause_change_of_strength', 'trigger_word': 'strengthen'}, {'Event_type': 'Catastrophe', 'trigger_word': 'hurricane'}, {'Event_type': 'Self_motion', 'trigger_word': 'peaked'}], 'id': 4},
{'sentence': 'A groundbreaking of the permanent memorial occurred in June 2006.', 'events': [{'Event_type': 'Social_event', 'trigger_word': 'memorial'}, {'Event_type': 'Presence', 'trigger_word': 'occurred'}], 'id': 5},
{'sentence': 'As winners, Chelsea took part in the 2012 UEFA Super Cup, losing 4–1 to Atlético Madrid, the winners of the 2011–12 UEFA Europa League.', 'events': [{'Event_type': 'Earnings_and_losses', 'trigger_word': 'losing'}], 'id': 6},
{'sentence': 'Two people died in the incident: the pilot, Pete Barnes, 50, and a pedestrian, Matthew Wood, 39, from Sutton in south London.', 'events': [{'Event_type': 'Death', 'trigger_word': 'died'}], 'id': 7},
{'sentence': 'A series of meetings with the Conservatives began shortly after the hung parliament was announced, and continued over the weekend after the election.', 'events': [{'Event_type': 'Process_start', 'trigger_word': 'began'}, {'Event_type': 'Change_of_leadership', 'trigger_word': 'election'}, {'Event_type': 'Expressing_publicly', 'trigger_word': 'announced'}, {'Event_type': 'Social_event', 'trigger_word': 'meetings'}], 'id': 8},
{'sentence': "The festival's acts come from a wide range of genres, such as: electro, rock, drum and bass, pop, R&B, reggae, house, punk, hardcore, metal, hip-hop, indie, techno, and more.", 'events': [{'Event_type': 'Social_event', 'trigger_word': 'festival'}, {'Event_type': 'Social_event', 'trigger_word': 'pop'}], 'id': 9},
{'sentence': 'The chase was short, as "Duguay Trouin" was a poor sailor with many of the crew sick and unable to report for duty.', 'events': [{'Event_type': 'Self_motion', 'trigger_word': 'chase'}], 'id': 10},
{'sentence': 'The ICTY convicted two JNA officers in connection with the massacre, and also tried former Serbian President Slobodan Milošević for a number of war crimes, including those committed at Vukovar.', 'events': [{'Event_type': 'Legal_rulings', 'trigger_word': 'convicted'}], 'id': 11},
{'sentence': 'Weakening as it drifted inland, Winifred persisted as a tropical depression for another five days after landfall before finally dissipating on 5 February.', 'events': [{'Event_type': 'Motion', 'trigger_word': 'drifted'}, {'Event_type': 'Cause_change_of_strength', 'trigger_word': 'Weakening'}, {'Event_type': 'Arriving', 'trigger_word': 'landfall'}, {'Event_type': 'Wearing', 'trigger_word': 'persisted'}, {'Event_type': 'Removing', 'trigger_word': 'dissipating'}], 'id': 12},
{'sentence': 'The whole community greets the Sun as they listen to Tibetan chants and guest musicians on the grassy hill.', 'events': [{'Event_type': 'Perception_active', 'trigger_word': 'listen'}, {'Event_type': 'Traveling', 'trigger_word': 'guest'}], 'id': 13},
{'sentence': 'In Libya the Islamic State of Iraq and the Levant (ISIL) has been able to control some limited territory in the ongoing civil war since 2014, amid allegations of local collaboration between the otherwise rivalling AQIM and ISIL.', 'events': [{'Event_type': 'Control', 'trigger_word': 'control'}, {'Event_type': 'Limiting', 'trigger_word': 'limited'}, {'Event_type': 'Hostile_encounter', 'trigger_word': 'war'}], 'id': 14},
{'sentence': 'The Inter-Provincial Series has been funded at least partly by the ICC via their TAPP programme.', 'events': [{'Event_type': 'Supply', 'trigger_word': 'funded'}], 'id': 15},
{'sentence': 'The "Struma" disaster joined that of SS "Patria" – sunk after Haganah sabotage while laden with Jewish refugees 15 months earlier – as rallying points for the Irgun and Lehi revisionist Zionist clandestine movements, encouraging their violent revolt against the British presence in Palestine.', 'events': [{'Event_type': 'Change_of_leadership', 'trigger_word': 'revolt'}, {'Event_type': 'Becoming_a_member', 'trigger_word': 'joined'}, {'Event_type': 'Self_motion', 'trigger_word': 'sunk'}, {'Event_type': 'Bringing', 'trigger_word': 'laden'}], 'id': 16},
{'sentence': 'On Friday, 28 February 1986, at 23:21 CET (22:21 UTC), Olof Palme, Prime Minister of Sweden, was fatally wounded by a single gunshot while walking home from a cinema with his wife Lisbet Palme on the central Stockholm street Sveavägen.', 'events': [{'Event_type': 'Bodily_harm', 'trigger_word': 'wounded'}, {'Event_type': 'Use_firearm', 'trigger_word': 'gunshot'}, {'Event_type': 'Self_motion', 'trigger_word': 'walking'}], 'id': 17},
{'sentence': 'The Department of Social Security (DSS) sent employees to receive claims for damage, requests for financial aid, and filings for unemployment benefits.', 'events': [{'Event_type': 'Sending', 'trigger_word': 'sent'}, {'Event_type': 'Request', 'trigger_word': 'requests'}, {'Event_type': 'Receiving', 'trigger_word': 'receive'}, {'Event_type': 'Assistance', 'trigger_word': 'aid'}, {'Event_type': 'Employment', 'trigger_word': 'employees'}, {'Event_type': 'Earnings_and_losses', 'trigger_word': 'benefits'}, {'Event_type': 'Damaging', 'trigger_word': 'damage'}], 'id': 18},
{'sentence': 'Air Algérie Flight 6289 (AH6289), was a domestic passenger flight which crashed on 6 March 2003, at the Aguenar – Hadj Bey Akhamok Airport in Algeria, killing all but one of the 103 people on board.', 'events': [{'Event_type': 'Catastrophe', 'trigger_word': 'crashed'}, {'Event_type': 'Killing', 'trigger_word': 'killing'}], 'id': 19},
{'sentence': "The accident is the RAF's worst peacetime disaster.", 'events': [{'Event_type': 'Catastrophe', 'trigger_word': 'accident'}], 'id': 20},
{'sentence': 'The governing Labour administration led by Gordon Brown was defeated in the election and lost its overall majority after 13 years in office.', 'events': [{'Event_type': 'Change_of_leadership', 'trigger_word': 'election'}, {'Event_type': 'Earnings_and_losses', 'trigger_word': 'lost'}, {'Event_type': 'Hostile_encounter', 'trigger_word': 'defeated'}], 'id': 21},
{'sentence': 'The battle is mentioned in the Babylonian Chronicles, now housed in the British Museum.', 'events': [{'Event_type': 'Hostile_encounter', 'trigger_word': 'battle'}, {'Event_type': 'Statement', 'trigger_word': 'mentioned'}, {'Event_type': 'Containing', 'trigger_word': 'housed'}], 'id': 22},
{'sentence': "Nicolas Anelka of France scored the first goal in Club World Cup history, while Brazilian champions Corinthians' goalkeeper Dida posted the first official clean sheet in the tournament.", 'events': [{'Event_type': 'Getting', 'trigger_word': 'scored'}, {'Event_type': 'Competition', 'trigger_word': 'champions'}, {'Event_type': 'Competition', 'trigger_word': 'tournament'}, {'Event_type': 'Sending', 'trigger_word': 'posted'}], 'id': 23},
{'sentence': 'Hawthorn, who were competing in their inaugural VFL Grand Final despite being in the competition since 1925, came into the game as minor premiers and favourites.', 'events': [{'Event_type': 'Competition', 'trigger_word': 'competing'}, {'Event_type': 'Motion', 'trigger_word': 'came'}], 'id': 24},
{'sentence': 'At the same time, General Joaquín Fanjul, commander of the military garrison based in Montaña barracks in Madrid, was preparing to launch the military rebellion in the city.', 'events': [{'Event_type': 'Process_start', 'trigger_word': 'launch'}, {'Event_type': 'GetReady', 'trigger_word': 'preparing'}], 'id': 25},
{'sentence': 'The Engineers were said to have missed their best back, Lieut.', 'events': [{'Event_type': 'Communication', 'trigger_word': 'said'}, {'Event_type': 'Earnings_and_losses', 'trigger_word': 'missed'}], 'id': 26},
{'sentence': 'They were organised on the initiative of deaf Frenchman Eugène Rubens-Alcais, who, just after the Games, co-founded the Comité International des Sports des Sourds with other "deaf sporting leaders".', 'events': [{'Event_type': 'Arranging', 'trigger_word': 'organised'}, {'Event_type': 'Collaboration', 'trigger_word': 'co-founded'}], 'id': 27},
{'sentence': 'The impeachment trial was formally opened on November 20, with twenty-one senators taking their oaths as judges, and Supreme Court Chief Justice Hilario Davide, Jr. presiding.', 'events': [{'Event_type': 'Openness', 'trigger_word': 'opened'}], 'id': 28},
{'sentence': 'The defense of Sihang Warehouse took place from October 26 to November 1, 1937, and marked the beginning of the end of the three-month Battle of Shanghai in the opening phase of the Second Sino-Japanese War.', 'events': [{'Event_type': 'Defending', 'trigger_word': 'defense'}, {'Event_type': 'Process_start', 'trigger_word': 'beginning'}, {'Event_type': 'Process_start', 'trigger_word': 'took place'}, {'Event_type': 'Recording', 'trigger_word': 'marked'}, {'Event_type': 'Process_end', 'trigger_word': 'end'}, {'Event_type': 'Hostile_encounter', 'trigger_word': 'Battle'}, {'Event_type': 'Hostile_encounter', 'trigger_word': 'War'}, {'Event_type': 'Openness', 'trigger_word': 'opening'}], 'id': 29},
{'sentence': 'The attack failed, and a small raid that evening inflicted little damage.', 'events': [{'Event_type': 'Attack', 'trigger_word': 'attack'}, {'Event_type': 'Agree_or_refuse_to_act', 'trigger_word': 'failed'}, {'Event_type': 'Causation', 'trigger_word': 'inflicted'}], 'id': 30},
{'sentence': "The report cites research that shows people's feelings about a police encounter depend significantly on whether they feel the officer displays respect and courtesy.", 'events': [{'Event_type': 'Change_sentiment', 'trigger_word': 'feel'}, {'Event_type': 'Adducing', 'trigger_word': 'cites'}, {'Event_type': 'Influence', 'trigger_word': 'depend'}], 'id': 31},
{'sentence': 'Fighting continued for the next two days.', 'events': [{'Event_type': 'Hostile_encounter', 'trigger_word': 'Fighting'}], 'id': 32},
{'sentence': 'On 19 March, he opened fire at the Ozar Hatorah Jewish day school in Toulouse, killing a rabbi and three children, also wounding four others.', 'events': [{'Event_type': 'Openness', 'trigger_word': 'opened'}, {'Event_type': 'Killing', 'trigger_word': 'killing'}], 'id': 33},
{'sentence': 'Peace negotiations and foreign involvement led to a ceasefire in 1995 that was broken the next year before a final peace agreement and new national elections were held in 1997.', 'events': [{'Event_type': 'Communication', 'trigger_word': 'negotiations'}, {'Event_type': 'Participation', 'trigger_word': 'involvement'}, {'Event_type': 'Process_end', 'trigger_word': 'ceasefire'}, {'Event_type': 'Sign_agreement', 'trigger_word': 'agreement'}, {'Event_type': 'Change_of_leadership', 'trigger_word': 'elections'}, {'Event_type': 'Bodily_harm', 'trigger_word': 'broken'}, {'Event_type': 'Causation', 'trigger_word': 'led to'}], 'id': 34},
{'sentence': 'On 2 August 1990 the Iraqi Army invaded and occupied Kuwait, which was met with international condemnation and brought immediate economic sanctions against Iraq by members of the UN Security Council.', 'events': [{'Event_type': 'Bringing', 'trigger_word': 'brought'}, {'Event_type': 'Attack', 'trigger_word': 'invaded'}, {'Event_type': 'Control', 'trigger_word': 'occupied'}, {'Event_type': 'Come_together', 'trigger_word': 'met with'}], 'id': 35},
{'sentence': 'The Fall of Philadelphia marked the fall of Philadelphia, the last independent Christian Greek settlement in western Asia Minor, to the Muslim Turks of the Ottoman Sultanate.', 'events': [{'Event_type': 'Recording', 'trigger_word': 'marked'}, {'Event_type': 'Motion_directional', 'trigger_word': 'fall'}], 'id': 36},
{'sentence': "They see the Council's decision as part of a wider 'cultural war' against 'Britishness' in Northern Ireland.", 'events': [{'Event_type': 'Deciding', 'trigger_word': 'decision'}, {'Event_type': 'Military_operation', 'trigger_word': 'war'}], 'id': 37},
{'sentence': 'In the wake of the Polish advance eastward, the Soviets sued for peace and the war ended with a cease-fire in October 1920.', 'events': [{'Event_type': 'Process_end', 'trigger_word': 'ended'}, {'Event_type': 'Cause_to_make_progress', 'trigger_word': 'advance'}, {'Event_type': 'Hostile_encounter', 'trigger_word': 'war'}, {'Event_type': 'Request', 'trigger_word': 'sued'}], 'id': 38},
{'sentence': 'Western fears of Soviet troops arriving at the German frontiers increased the interest of Western powers in the war.', 'events': [{'Event_type': 'Cause_change_of_position_on_a_scale', 'trigger_word': 'increased'}, {'Event_type': 'Arriving', 'trigger_word': 'arriving at'}, {'Event_type': 'Hostile_encounter', 'trigger_word': 'war'}], 'id': 39},
]



def gpt_get_completion(prompt, gpt_llm, model="gpt-3.5-turbo-0125"):
    """
    Generate completions using the OpenAI GPT language model.

    Parameters:
        prompt (str): The input prompt for the model.
        gpt_llm: An instance of the OpenAI API with the appropriate API key.
        model (str): The name of the GPT model to use for completion.
                     Default is "gpt-3.5-turbo-0125".

    Returns:
        str: The completion generated by the model.
    """
    completion = gpt_llm.chat.completions.create(
    model= model,
      messages=[
        {"role": "system", "content": "Act as a perfect triple extractor model"},
        {"role": "user", "content": prompt}
      ],
      temperature=0
    )
    return(completion.choices[0].message.content)


def llama_prompt_run(model, prompt, headers):
  """
    Send a prompt to the LLaMA  API for processing.

    Parameters:
        model (str): The model endpoint to send the prompt to.
        prompt (str): The input prompt for the model.
        headers: HTTP headers required for authorization.

    Returns:
        dict: A JSON response containing the output generated by the model.
  """
  input = {
    "messages": [
      { "role": "system", "content": "You are an expert at extracting event from given sentences for exvent extraction" },
      { "role": "user", "content": prompt }
    ]
  }
  response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
  return response.json()


def EE_prompt_runner(prompt_base, dataset, list_of_events, Num_samples=20, CoT = False, model=None, API = None, sub_model = 'gpt-3.5-turbo-0125'):
    """
    Run the prompt-based event extraction process.

    Parameters:
        prompt_base (str): The base prompt for event extraction.
        dataset (list): A list of dictionaries representing the dataset containing sentences.
        list_of_events (list): A list of event types to be extracted.
        Num_samples (int): The number of samples to process. Default is 20.
        CoT (bool): Whether to use "Chain of Thought" reasoning. Default is False.
        model (str): The name of the model to use for event extraction.
        API: The API key required for model access.
        sub_model (str): The specific GPT sub-model to use. Default is 'gpt-3.5-turbo-0125'.

    Returns:
        tuple: A tuple containing the output list and, if CoT is enabled, the reasoning list.
    """
    if model == 'Gemini':
      
      gemini_api_key = API
      os.environ["GOOGLE_API_KEY"] = gemini_api_key
      gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    elif model == 'GPT':
      gpt_api_key = API
      gpt_llm = OpenAI(api_key=gpt_api_key)

    elif model == 'LLaMA':
      headers = {"Authorization": "Bearer {}".format(API)}


    counter = 0
    output_list = []
    reasoning_list = []
    for idx, sample in enumerate(dataset):
        prompt = prompt_base.format(list_of_events, sample['sentence'])

        try:
            # answer contain the model response to our prompt
            if model == 'GPT':
                answer = gpt_get_completion(prompt, gpt_llm, model = sub_model )
                
            elif model == 'LLaMA':
                prediction = llama_prompt_run("@hf/thebloke/llama-2-13b-chat-awq", prompt, headers=headers)
                answer = prediction['result']['response']
            else:
                prediction = gemini_llm.invoke(prompt)
                answer = prediction.content

            if CoT == 'CoT': # Collect the entire response of LLM for CoT 
                reasoning_list.append(answer)

            # print(answer)
            if "Event type: " not in answer: # Help parser to extract model response eficiently. 
                answer = "Event type: " + answer
            output_list.append({"id": sample["id"], 'prediction': answer})
            counter += 1
        except Exception as e:
            # Some sentence may be blocked by Gemini for some safty settings
            print('error for sentence: ', sample['sentence'])
        if counter == Num_samples:
            break
    if  CoT == 'CoT':
        return output_list, reasoning_list
    return output_list, None


def EE_output_processing(org_output):
  """
    Process the raw output from event extraction to a structured format.

    Parameters:
        org_output (list): The original output list from event extraction.

    Returns:
        list: A processed list containing dictionaries with structured information. {id : ['event_type', 'event_type']}
  """
  output = copy.deepcopy(org_output)
  # remove empty list samples
  output = [item for item in output if len(item.keys())>= 2]
  for sample in output:
    sample['prediction'] = [item.strip().replace('.', '') for item in sample['prediction'].split("Event type: ")[1].strip().split('\n')[0].split(',')]
  return output

# for sample in output:
#     l = []
#     for item in sample['prediction'].split("Event type: ")[1].strip().split(','):
#       if '\n' in item:
#         item = item.split('\n')[0].strip()
#       l.append(item)
#     sample['prediction'] = l

def EE_gold_samples_generator(MAVAN_dataset):
  """
    Generate gold samples from a dataset.

    Parameters:
        MAVAN_dataset (list): The dataset containing samples with event annotations.

    Returns:
        dict: A dictionary mapping sample IDs to lists of gold event types.
  """
  gold_samples = {}
  for sample in MAVAN_dataset:
    gold_samples[sample['id']] = [event['Event_type'] for event in  sample['events']]
  return gold_samples


def reorder_lists(list1, list2):
    """
    Reorder two lists to align elements and ensure consistency.

    Parameters:
        list1 (list): The first list to be reordered.
        list2 (list): The second list to be reordered.

    Returns:
        tuple: Two reordered lists with aligned elements.
    """
  
    # Combine the input lists and find unique elements
    merged_list = list1 + list2
    unique_elements = list(set(merged_list))

    # Create placeholders for elements not present in each list
    reordered_list1 = []
    reordered_list2 = []
    for element in unique_elements:
        if element in list1 and element in list2:
            reordered_list1.append(element)
            reordered_list2.append(element)
        elif element in list1:
            reordered_list1.append(element)
            reordered_list2.append("0")
        elif element in list2:
            reordered_list1.append("0")
            reordered_list2.append(element)

    return reordered_list1, reordered_list2


def EE_evaluation(gold_samples, processed_output):
  """
    Generate a final list of labels and predictions which will be used in metric calculation.

    Parameters:
        gold_samples (dict): A dictionary of gold samples.
        processed_output (list): The processed output from event extraction.

    Returns:
        tuple: Two lists containing labels and predictions.
  """
  labels = []
  predictions = []

  for item in processed_output:
    gold, pred = reorder_lists(gold_samples[item['id']] ,item['prediction'])
    labels.extend(gold)
    predictions.extend(pred)

  return labels, predictions


def EE_metric_calculation(labels, predictions, event_labels):
  """
  When predicting an event not present in the ground truth, our predictions are incorrect, resulting in a decrease in our Precision (P).
  If we fail to predict any event when one exists in the ground truth, it decreases our Recall (R).
  This is why we include zeros in both the ground truth and prediction lists, to address the inconsistency in the number of extracted events."
  """
  micro_p = round(precision_score(labels,predictions, labels = event_labels,average='micro')*100.0, 2)
  micro_r = round(recall_score(labels,predictions, labels = event_labels, average='micro')*100.0, 2)
  micro_f1 = round(f1_score(labels,predictions, labels = event_labels, average='micro')*100.0, 2)


  print("Micro_F1:", round(micro_f1,2))
  print("Micro_Precision:", round(micro_p,2))
  print("Micro_Recall:", round(micro_r,2))

  return micro_p, micro_r, micro_f1


def EE_print_output(output, dataset, dataset_gold_samples, cot_reasoning_list = None):
  """
    Print the output of event extraction for analysis.

    Parameters:
        output (list): The output list from event extraction.
        dataset (list): The dataset containing samples.
        dataset_gold_samples (dict): The gold samples extracted from the dataset.
        cot_reasoning_list (list): A list containing Chain of Thought reasoning.
  """
  index = 0
  abstained_counter = 0
  sample_num = 0
  for item in output:
    if len(item['prediction']) == 0:
      abstained_counter += 1
  print("Rate of unstructured or abstained output: {}%".format(abstained_counter/len(output)))
  print()
  for i, item in enumerate(output):
    # sample_num += 1
    for smple in dataset:
      if smple['id'] == item['id']:
        print("Sentence: ", smple['sentence'])
        print("Gold -----> ", smple['events'] )
        print()
    if cot_reasoning_list:
      print("Reasoning ---> ", cot_reasoning_list[i])
      print()
    print(item['prediction'])
    print("\n***\n")


def EE_write_output(output, dataset, dataset_gold_samples,  prompt_type, target_prompt, resutls, model, cot_reasoning_list=None):
    """
    Write the output of event extraction to a file.

    Parameters:
        output (list): The processed output from event extraction.
        dataset (list): The dataset containing samples.
        dataset_gold_samples (dict): The gold samples extracted from the dataset.
        prompt_type (str): The type of prompt used for event extraction.
        target_prompt (str): The target prompt used.
        results (tuple): Evaluation results (micro-precision, micro-recall, micro-F1).
        model (str): The name of the model used.
        cot_reasoning_list (list): A list containing Chain of Thought reasoning.
    """
    target_dir = './MAVAN_Experiments_Log/{}'.format(model)
    try:
      create_directory(target_dir)
    except Exception as e:
      print(e)
    index = 0
    abstained_counter = 0
    sample_num = 0
    with open('{}/{}.txt'.format(target_dir, prompt_type), "w") as file:
        for item in output:
            if len(item['prediction']) == 0:
                abstained_counter += 1
        file.write("Prompt: \n''' " + target_prompt + "''' \n")
        micro_p, micro_r, micro_f1  = resutls
        file.write("micro_p = {}\nmicro_r = {}\nmicro_f1 = {}\n\n".format(micro_p, micro_r, micro_f1 ))
        file.write("Rate of unstructured or abstained output: {}%\n\n".format(abstained_counter / len(output)))
        for i, item in enumerate(output):
            for smple in dataset:
                if smple['id'] == item['id']:
                    file.write("Sentence: {}\n".format(smple['sentence']))
                    file.write("Gold -----> {}\n".format(smple['events']))
                    file.write("\n")
            if cot_reasoning_list:
                file.write("Reasoning ---> {}\n\n".format(cot_reasoning_list[i]))
            file.write("{}\n".format(item['prediction']))
            file.write("\n***\n\n")


def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print("Directory '{}' created successfully.".format(directory_name))
    except FileExistsError:
        print("Directory '{}' already exists.".format(directory_name))
    except Exception as e:
        print("An error occurred:", e)

def event_list_extractor(MAVEN_dataset_40samples):
    
    sample_list_of_events = []
    for item in MAVEN_dataset_40samples:
        for event in item['events']:
            sample_list_of_events.append(event['Event_type'])
    sample_list_of_events = list(set(sample_list_of_events))
    return sample_list_of_events


    
flag = 1
if __name__ == "__main__":
  args = sys.argv
  while flag:
    if len(args) < 4:
        print("Please provide at least prompt, model, and API.")
        
    elif len(args) == 4:
        prompt_type, model, API = args[1:]
        sub_model = 'gpt-3.5-turbo-0125'
        flag = 0
        
    else:
        prompt_type, model, API = args[1:4]  # Take only the first three items
        sub_model = args[4]  # Additional items beyond prompt, model, and API
        flag = 0

  try:
    create_directory('MAVAN_Experiments_Log')
  except Exception as e:
    print(e)
  
  sample_list_of_events = event_list_extractor(MAVEN_dataset_40samples)
  gold_samples = EE_gold_samples_generator(MAVEN_dataset_40samples)
  Num_samples = 2

  if prompt_type == 'zero_shot':
      target_prompt = zero_shot_prompt
  elif prompt_type == 'few_shot':
      target_prompt = few_shot_prompt
  elif  prompt_type == 'CoT':
      target_prompt = CoT_prompt

  output, reasoning_list = EE_prompt_runner(target_prompt, MAVEN_dataset_40samples, sample_list_of_events, Num_samples = Num_samples, CoT = prompt_type , model = model, API = API, sub_model=sub_model )    
  processed_output = EE_output_processing(output)
  labels, predictions = EE_evaluation(gold_samples, processed_output)
  results = EE_metric_calculation(labels, predictions, sample_list_of_events)
  EE_print_output(output, MAVEN_dataset_40samples, gold_samples, cot_reasoning_list = reasoning_list)
  
  if len(args) == 4:
    EE_write_output(processed_output, MAVEN_dataset_40samples, gold_samples, prompt_type, target_prompt, results, model, cot_reasoning_list = reasoning_list )
  else:
    EE_write_output(processed_output, MAVEN_dataset_40samples, gold_samples, prompt_type, target_prompt, results, sub_model, cot_reasoning_list = reasoning_list )