import sys
from triple_extraction_utils import *


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/375121881c6ec963b21944159717e15e/ai/run/"


zero_shot_prompt = '''
The list of predicates: {}
What Subject-Predicate-Object triples are included in the following sentence? Please return the possible answers according to the list above. Require the answer only in the form: [subject, predicate, object]
The given sentence is : {}
Triples: '''


few_shot_prompt = '''
Given a sentence, Subject-Predicate-Object triples are included in the following sentence using predicates from the following list {}?
Please return the answers only in the form: Triples: [subject, predicate, object]

Examples of task are below:

Example 1:
The given sentence is : ALERT ¥ US missionary Laura Silsby freed in Haiti : lawyer
Triples: [Laura Silsby, per:title, missionary]

Example 2:
The given sentence is : An optimistic outlook from troubled US mortgage giant Countrywide Financial on Friday , coupled with expectations for a US rate cut on Wednesday , raised hopes that the fallout from the US housing woes is easing , dealers said .
Triples: [Countrywide Financial, org:country_of_branch, US]

Example 3:
The given sentence is : `` I learn from students and I challenge them , '' says Heloise , 58 , who took over the family hints business when her mother , also named Heloise , died in 1977 .
Triples: [Heloise, per:parents, Heloise]

Example 4:
The given sentence is : In the same hearing , three former Chongqing police officers -- Huang Daiqiang , Zhao Liming and Chen Tao -- received jail terms ranging from 17 and a half years to 20 years after being convicted on corruption charges .
Triples: [Chen Tao, per:employee_of, Chongqing]

Example 5:
The given sentence is : Neighbors described 48-year-old George Sodini as anti-social , and the Web page in his name showcased a resume setting forth his credentials as an unhappy loner .
Triples: [George Sodini, per:age, 48-year-old]

Example 6:
The given sentence is : In the year her American husband has been detained in Cuba , accused of spying for the US , Judy Gross has been forced to sell the family home and move into a small apartment in Washington .
Triples: [Gross, per:countries_of_residence, US]

Example 7:
The given sentence is : But for her , the benefits she got from Scientology still outweighed all that .
Triples: [she, per:religion, Scientology]

Example 8:
The given sentence is : A former Pakistani lawmaker has been arrested in connection with the Taliban kidnapping and murder of a Polish engineer , the Polish foreign ministry said Sunday .
Triples: [lawmaker, per:origin, Pakistani]

Example 9:
The given sentence is : It sold ALICO to MetLife Inc for $ 162 billion .
Triples: [ALICO, org:member_of, MetLife]

Example 10:
The given sentence is : No one knows how Tamaihia Lynae Moore died , but the foster mother of the Sacramento toddler has been arrested for her murder .
Triples: [her, per:cities_of_residence, Sacramento]

Now you try this task:
The given sentence is : {}
Triples:
'''

CoT_prompt = '''

Given a sentence, identify and extract relevant triples of entities connected by a predicate from the provided list.
The format of each triple should be [head, predicate, tail], separated by commas.

Instructions:

Step (1): Entity Extraction
- Extract the entities from the given sentence.
- Group sequences of words that indicate the same role in the sentence as a single entity.
- Exclude adjectives from entities to focus on core nouns.

Step (2): Triple Identification
- Identify a single relevant predicate from the given list that best describes the relationship between the entities. List of predicates: {}.
- Ensure that the order of subject, predicate, and object is maintained based on the predicate's meaning.
- Consider the semantic relationship between entities when selecting a predicate.

Step (3): Triple Formation
- Construct triples in the format [head, predicate, tail].
- Prioritize the triples based on your confidence in their relevance and accuracy.
- Return up to two of the most confident triples. If there is only one confident triple, return that singular triple. Ensure that the predicates in each triple are distinct.


Exmaple 1:
The given sentence is : John Graham , a 55-year-old man from Canada , is accused of shooting Aquash in the head and leaving her to die on the Pine Ridge reservation in South Dakota .

Step (1): Entities: 'Josh Graham', '55', 'Canada', 'Aquah', 'Pine Ridge reservation', 'South Dakota'

Step (2):
'Josh Graham' - 'from' - 'Canada'
'Josh Graham' - 'year-old' - '55'
'Aquash' - 'leaving her to die' - 'South Dakota'

Match possible predicates on based on the previously given list of predicates:

'Josh Graham' - 'per:countries_of_residence' - 'Canada'
'Josh Graham' - 'per:age' - '55'
'Aquash' - 'per:stateorprovince_of_death' - 'South Dakota'

Step (3):
Triples:
[Josh Graham, per:countries_of_residence, Canada]
[Aquash, per:stateorprovince_of_death, South Dakota]

Example 2:
The given sentence is : Salmaan Taseer , the governor of Punjab province -- where Bibi has been held in jail for more than a year -- said he had forwarded a petition presenting the facts of the case to President Asif Ali Zardari on Monday .

Step (1): Entities: 'Salmaan Taseer', 'governor', 'Punjab', 'Bibi', 'President', 'Asif Ali Zardari'

Step (2):
'Salmaan Taseer' - 'the' - 'governor'
'President' - - 'Asif Ali Zardari'
'Punjab' - 'where has been held in jail' - 'Bibi'

Match possible predicates on based on the previously given list of predicates:

'Salmaan Taseer' - 'per:title' - 'governor'
'Asif Ali Zardari' - 'per:title' - 'President'
'Bibi' - 'per:stateorprovinces_of_residence' - 'Punjab'

Step (3):
Triples: 
[Bibi, per:stateorprovinces_of_residence, Punjab]
[Asif Ali Zardari, per:title, President]

Example 3:
The given sentence is : Her brother-in-law , Wen , served as a top Chongqing police official for 16 years before taking over the city 's judiciary .

Step (1): Entities: 'Her', 'brother-in-law', 'Wen', 'Chongqing', 'police official'

Step (2):
'Her' - 'brother-in-law' - 'Wen'
'Wen' - 'served as' - 'Chongqing'
'Wen' - 'served as' - 'police official'

Match possible predicates on based on the previously given list of predicates:

'Her' - 'per:other_family' - 'Wen'
'Wen' - 'per:employee_of' - 'Chongqing'
'Wen' - 'per:employee_of' - 'police official'

Step (3):
Triples: 
[Her, per:other_family, Wen]
[Wen, per:employee_of, Chongqing]

Now you try this task with the following:
The given sentence is : {}
Triples:
'''

RE_TACRED_dataset_40samples =[
{'id': 2018, 'relation': 'per:city_of_birth', 'tokens': 'Andrew E. Lange was born in Urbana , Ill. , on July 23 , 1957 , the oldest son of Joan Lange , a school librarian , and Albert Lange , an architect , and grew up in Easton , Conn. .', 'h': {'name': 'Andrew E. Lange', 'pos': [0, 3]}, 'subj_type': 'PERSON', 't': {'name': 'Urbana', 'pos': [6, 7]}, 'obj_type': 'CITY'} ,
{'id': 335, 'relation': 'org:member_of', 'tokens': 'Bank of America Corp. is taking a similar approach with newly acquired Countrywide Financial Corp. as part of an $ 8.4 billion , 12-state legal settlement reached this month .', 'h': {'name': 'Countrywide Financial Corp.', 'pos': [12, 15]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'Bank', 'pos': [0, 1]}, 'obj_type': 'ORGANIZATION'} ,
{'id': 2387, 'relation': 'per:stateorprovince_of_birth', 'tokens': 'Although her family was from Arkansas , she was born in Washington state , where her father was working on a construction project .', 'h': {'name': 'she', 'pos': [7, 8]}, 'subj_type': 'PERSON', 't': {'name': 'Washington', 'pos': [11, 12]}, 'obj_type': 'STATE_OR_PROVINCE'} ,
{'id': 416, 'relation': 'org:city_of_branch', 'tokens': "Even at its peak , in 2007 , ALICO 's portfolio of credit-default swaps was just a fraction of the one at AIG Financial Products , the London shop whose collapsing business led the U.S. government to prop up AIG , the biggest bailout in American history .", 'h': {'name': 'ALICO', 'pos': [8, 9]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'London', 'pos': [27, 28]}, 'obj_type': 'CITY'} ,
{'id': 199, 'relation': 'per:children', 'tokens': 'But the 33-year-old daughter of the late civil rights attorney William Kunstler , co-director of a documentary film about her father that opened Friday in the Bay Area , says one lesson everyone should have learned from Central Park is not to rush to judgment .', 'h': {'name': 'William Kunstler', 'pos': [10, 12]}, 'subj_type': 'PERSON', 't': {'name': 'her', 'pos': [19, 20]}, 'obj_type': 'PERSON'} ,
{'id': 246, 'relation': 'org:founded_by', 'tokens': "`` Last-minute shoppers swamped stores over the weekend , allowing retailers to breathe a sigh of relief , '' said Bill Martin , co - founder of ShopperTrak .", 'h': {'name': 'ShopperTrak', 'pos': [27, 28]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'Bill', 'pos': [20, 21]}, 'obj_type': 'PERSON'} ,
{'id': 66, 'relation': 'per:employee_of', 'tokens': "Ivory Coast 's new U.N. ambassador , Youssoufou Bamba , said he is worried about his country 's future and is consulting with members of the Security Council ahead of a meeting next week on ways to help Ouattara assume power .", 'h': {'name': 'Youssoufou Bamba', 'pos': [7, 9]}, 'subj_type': 'PERSON', 't': {'name': 'Ivory', 'pos': [0, 1]}, 'obj_type': 'COUNTRY'} ,
{'id': 1026, 'relation': 'per:schools_attended', 'tokens': 'He graduated from Muhlenberg College in Allentown , Pa. , and received a Ph.D. in pharmacology from Temple University .', 'h': {'name': 'He', 'pos': [0, 1]}, 'subj_type': 'PERSON', 't': {'name': 'Muhlenberg', 'pos': [3, 4]}, 'obj_type': 'ORGANIZATION'} ,
{'id': 2, 'relation': 'no_relation', 'tokens': 'Eugenio Vagni , the Italian worker of the ICRC , and colleagues Andreas Notter of Switzerland and Mary Jean Lacaba of the Philippines were released by their Abu Sayyaf captors separately .', 'h': {'name': 'Eugenio Vagni', 'pos': [0, 2]}, 'subj_type': 'PERSON', 't': {'name': 'Switzerland', 'pos': [15, 16]}, 'obj_type': 'COUNTRY'} ,
{'id': 908, 'relation': 'org:number_of_employees/members', 'tokens': 'The 5,000-member High Point Church was founded in 2000 by Simons and his wife , April , whose brother is Joel Osteen , well-known pastor of the 38,000-member Lakewood Church in Houston .', 'h': {'name': 'High Point Church', 'pos': [3, 6]}, 'subj_type': 'ORGANIZATION', 't': {'name': '5,000-member', 'pos': [2, 3]}, 'obj_type': 'NUMBER'} ,
{'id': 1854, 'relation': 'org:founded', 'tokens': 'UASR was founded in 1989 by Mousa abu Mazook < http://wwwinvestigativeprojectorg/profile/106 > , who currently serves as the Deputy Chief of the HAMAS political bureau in Damascus , Syria , and has been listed as a Specially Designated Global Terrorist by the US government in 1995 .', 'h': {'name': 'UASR', 'pos': [0, 1]}, 'subj_type': 'ORGANIZATION', 't': {'name': '1989', 'pos': [4, 5]}, 'obj_type': 'DATE'} ,
{'id': 73, 'relation': 'per:age', 'tokens': 'She was 61 .', 'h': {'name': 'She', 'pos': [0, 1]}, 'subj_type': 'PERSON', 't': {'name': '61', 'pos': [2, 3]}, 'obj_type': 'NUMBER'} ,
{'id': 1100, 'relation': 'per:city_of_death', 'tokens': 'The body of Mario Gonzalez was found half buried in a house under construction in Chihuahua city after one of the suspects told officials where they could find him , federal police commissioner Facundo Rosas told a news conference .', 'h': {'name': 'Mario Gonzalez', 'pos': [3, 5]}, 'subj_type': 'PERSON', 't': {'name': 'Chihuahua', 'pos': [15, 16]}, 'obj_type': 'STATE_OR_PROVINCE'} ,
{'id': 638, 'relation': 'org:members', 'tokens': "A new distribution company , Tribeca Film , founded by the festival 's parent company Tribeca Enterprises , will make a dozen films -- including Whitecross ' directorial debut `` sex & drugs & rock & roll '' -- available on TV via video-on-demand in some 40 million homes .", 'h': {'name': 'Tribeca Enterprises', 'pos': [15, 17]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'Tribeca', 'pos': [5, 6]}, 'obj_type': 'ORGANIZATION'} ,
{'id': 183, 'relation': 'org:website', 'tokens': 'By Mike Blair , American Free Press 12/8/2003 http://www.americanfreepress.net/ The Great Depression is nothing compared with what in on the horizon .', 'h': {'name': 'American Free Press', 'pos': [4, 7]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'http://www.americanfreepress.net/', 'pos': [8, 9]}, 'obj_type': 'URL'} ,
{'id': 463, 'relation': 'org:stateorprovince_of_branch', 'tokens': "The greater relative reliance on share awards `` misaligned '' CEO and shareholder interests , according to a February report by the Corporate Library , a shareholder governance research firm in Portland , Maine .", 'h': {'name': 'Corporate Library', 'pos': [22, 24]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'Maine', 'pos': [33, 34]}, 'obj_type': 'STATE_OR_PROVINCE'} ,
{'id': 1545, 'relation': 'per:country_of_death', 'tokens': 'They say Vladimir Ladyzhenskiy died late Saturday during the Sauna World Championships in southern Finland , while his Finnish rival Timo Kaukonen was rushed to a hospital .', 'h': {'name': 'Vladimir Ladyzhenskiy', 'pos': [2, 4]}, 'subj_type': 'PERSON', 't': {'name': 'Finland', 'pos': [14, 15]}, 'obj_type': 'COUNTRY'} ,
{'id': 213, 'relation': 'org:alternate_names', 'tokens': "`` To fully reflect the will of the people in the 12th Five-Year Plan , advice and suggestions from all Chinese people are welcome in the following two months , '' Zhang Ping , director of the National Development and Reform Commission -LRB- NDRC -RRB- , said at a press conference in Beijing .", 'h': {'name': 'National Development and Reform Commission', 'pos': [37, 42]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'NDRC', 'pos': [43, 44]}, 'obj_type': 'ORGANIZATION'} ,
{'id': 336, 'relation': 'per:stateorprovince_of_death', 'tokens': 'Norris Church Mailer , who was the sixth and final wife of Pulitzer Prize-winning novelist Norman Mailer and was also a model , a painter and a respected author in her own right , died Nov 21 at her home in Brooklyn , NY .', 'h': {'name': 'Norris Church Mailer', 'pos': [0, 3]}, 'subj_type': 'PERSON', 't': {'name': 'NY', 'pos': [43, 44]}, 'obj_type': 'STATE_OR_PROVINCE'} ,
{'id': 193, 'relation': 'per:origin', 'tokens': "`` God wanted us to come here to help children , we are convinced of that , '' Laura Silsby , one of 10 Americans accused of trafficking Haitian children , said Monday through the bars of a jail cell here .", 'h': {'name': 'Laura Silsby', 'pos': [18, 20]}, 'subj_type': 'PERSON', 't': {'name': 'Americans', 'pos': [24, 25]}, 'obj_type': 'NATIONALITY'} ,
{'id': 9, 'relation': 'per:title', 'tokens': 'Lomax shares a story about Almena Lomax , his mother and a newspaper owner and journalist in Los Angeles , taking her family on the bus to Tuskegee , Ala. , in 1961 .', 'h': {'name': 'her', 'pos': [21, 22]}, 'subj_type': 'PERSON', 't': {'name': 'journalist', 'pos': [15, 16]}, 'obj_type': 'TITLE'} ,
{'id': 42, 'relation': 'org:top_members/employees', 'tokens': "`` There is a sense of pride in who he is , '' said Marc Morial , president of the National Urban League .", 'h': {'name': 'National Urban League', 'pos': [20, 23]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'Marc', 'pos': [14, 15]}, 'obj_type': 'PERSON'} ,
{'id': 532, 'relation': 'per:cause_of_death', 'tokens': "An autopsy concluded that Moore 's death was a homicide and investigators determined that Walker allegedly caused the baby 's death , Von Schoech said .", 'h': {'name': 'Moore', 'pos': [4, 5]}, 'subj_type': 'PERSON', 't': {'name': 'homicide', 'pos': [9, 10]}, 'obj_type': 'CAUSE_OF_DEATH'} ,
{'id': 124, 'relation': 'per:charges', 'tokens': "The verdict comes two days after Chongqing 's former police chief Wen Qiang went on trial for rape and accepting bribes in exchange for protecting the city 's extensive gang network .", 'h': {'name': 'Wen Qiang', 'pos': [11, 13]}, 'subj_type': 'PERSON', 't': {'name': 'rape', 'pos': [17, 18]}, 'obj_type': 'CRIMINAL_CHARGE'} ,
{'id': 153, 'relation': 'per:date_of_death', 'tokens': 'According to the teenager , Samudio was kidnapped in Rio in early June and killed some days later , in neighboring Minas Gerais state .', 'h': {'name': 'Samudio', 'pos': [5, 6]}, 'subj_type': 'PERSON', 't': {'name': 'days', 'pos': [16, 17]}, 'obj_type': 'DATE'} ,
{'id': 227, 'relation': 'per:religion', 'tokens': 'Judy Gross says he was working at a Jewish community center in Havana , helping Jewish groups on the island communicate with one another and get access to the Internet so they could look at Wikipedia and online prayer books .', 'h': {'name': 'he', 'pos': [3, 4]}, 'subj_type': 'PERSON', 't': {'name': 'Jewish', 'pos': [8, 9]}, 'obj_type': 'RELIGION'} ,
{'id': 5533, 'relation': 'org:dissolved', 'tokens': 'New Fabris closed down June 16 .', 'h': {'name': 'New Fabris', 'pos': [0, 2]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'June', 'pos': [4, 5]}, 'obj_type': 'DATE'} ,
{'id': 864, 'relation': 'per:spouse', 'tokens': "The U.S. government says Gross was in Cuba as part of a USAID program to distribute communications equipment to the island 's 1,500-strong Jewish community , and both the State Department and Gross 's wife , Judy , made fresh appeals this week for his release .", 'h': {'name': 'his', 'pos': [44, 45]}, 'subj_type': 'PERSON', 't': {'name': 'Judy', 'pos': [36, 37]}, 'obj_type': 'PERSON'} ,
{'id': 68, 'relation': 'per:countries_of_residence', 'tokens': 'Anna Mae Pictou Aquash , a Mi ` kmaq Indian from Canada , was brutally murdered in 1975 .', 'h': {'name': 'Anna Mae Pictou Aquash', 'pos': [0, 4]}, 'subj_type': 'PERSON', 't': {'name': 'Canada', 'pos': [11, 12]}, 'obj_type': 'COUNTRY'} ,
{'id': 77, 'relation': 'per:parents', 'tokens': 'The Dutch newspaper Brabants Dagblad said the boy was probably from Tilburg in the southern Netherlands and that he had been on safari in South Africa with his mother Trudy , 41 , father Patrick , 40 , and brother Enzo , 11 .', 'h': {'name': 'he', 'pos': [18, 19]}, 'subj_type': 'PERSON', 't': {'name': 'Patrick', 'pos': [34, 35]}, 'obj_type': 'PERSON'} ,
{'id': 1658, 'relation': 'org:shareholders', 'tokens': "COMMENT : Prachai is more in line with PAD , the anti-Thaksin protesters , as he was a major financier of theirs in a hope to get back his company TPI - you can read more about his company 's trouble here .", 'h': {'name': 'TPI', 'pos': [30, 31]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'he', 'pos': [15, 16]}, 'obj_type': 'PERSON'} ,
{'id': 712, 'relation': 'per:other_family', 'tokens': 'Xie was the sister-in-law of Wen Qiang , formerly the second in command of the Chongqing police and director of the justice bureau before he was arrested .', 'h': {'name': 'he', 'pos': [24, 25]}, 'subj_type': 'PERSON', 't': {'name': 'Xie', 'pos': [0, 1]}, 'obj_type': 'PERSON'} ,
{'id': 3027, 'relation': 'per:date_of_birth', 'tokens': 'Andrew E Lange was born in Urbana , Ill , on July 23 , 1957 , the oldest son of Joan Lange , a school librarian , and Albert Lange , an architect , and grew up in Easton , Conn .', 'h': {'name': 'Andrew E Lange', 'pos': [0, 3]}, 'subj_type': 'PERSON', 't': {'name': 'July', 'pos': [11, 12]}, 'obj_type': 'DATE'} ,
{'id': 373, 'relation': 'org:political/religious_affiliation', 'tokens': "The average chief executive of a company listed on the Standard & amp ; Poor 's 500-stock index earned $ 15.06 million in 2006 , according to the Corporate Library , an independent research firm .", 'h': {'name': 'Corporate Library', 'pos': [28, 30]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'independent', 'pos': [32, 33]}, 'obj_type': 'RELIGION'} ,
{'id': 398, 'relation': 'per:siblings', 'tokens': 'Dutch newspaper Babants Dagblad said the boy was likely Ruben van Assouw from Tilburg in the southern Netherlands who had been on safari in South Africa with his mother Trudy , 41 , father Patrick , 40 , and his brother Enzo , 11 .', 'h': {'name': 'Ruben van Assouw', 'pos': [9, 12]}, 'subj_type': 'PERSON', 't': {'name': 'Enzo', 'pos': [41, 42]}, 'obj_type': 'PERSON'} ,
{'id': 535, 'relation': 'per:stateorprovinces_of_residence', 'tokens': 'Clint Henry , the pastor at Central Valley Baptist Church in Meridian , Idaho , which Silsby and several other detainees attend , also called for Haiti to quickly free the group .', 'h': {'name': 'Silsby', 'pos': [16, 17]}, 'subj_type': 'PERSON', 't': {'name': 'Idaho', 'pos': [13, 14]}, 'obj_type': 'STATE_OR_PROVINCE'} ,
{'id': 378, 'relation': 'per:cities_of_residence', 'tokens': "There was just one problem : No mention was made of Alan P. Gross , an American from Potomac , Md. , who passed the holiday in a Cuban military facility , where he has been imprisoned for a year without trial because he tried to help Cuba 's Jews .", 'h': {'name': 'Alan P. Gross', 'pos': [11, 14]}, 'subj_type': 'PERSON', 't': {'name': 'Potomac', 'pos': [18, 19]}, 'obj_type': 'CITY'} ,
{'id': 12, 'relation': 'per:identity', 'tokens': "Against this background , Ouattara 's new United Nations ambassador Youssoufou Bamba gave a stark warning Wednesday as he received his credentials from UN Secretary-General Ban Ki-moon in New York .", 'h': {'name': 'his', 'pos': [20, 21]}, 'subj_type': 'PERSON', 't': {'name': 'he', 'pos': [18, 19]}, 'obj_type': 'PERSON'} ,
{'id': 107, 'relation': 'org:country_of_branch', 'tokens': 'AIG said it had transferred ownership to the Federal Reserve Bank of parts of two subsidiaries , ALICO which is active in life assurance in the United States and AIA which provides life assurance abroad .', 'h': {'name': 'ALICO', 'pos': [17, 18]}, 'subj_type': 'ORGANIZATION', 't': {'name': 'United', 'pos': [26, 27]}, 'obj_type': 'COUNTRY'} ,
{'id': 1, 'relation': 'no_relation', 'tokens': 'Messina Denaro has been trying to impose his power in Palermo , the Sicilian capital , and become the new head of the Sicilian Mafia , weakened by the arrest of Provenzano in April 2006 .', 'h': {'name': 'his', 'pos': [7, 8]}, 'subj_type': 'PERSON', 't': {'name': 'Palermo', 'pos': [10, 11]}, 'obj_type': 'CITY'}
]

    
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
    create_directory('RE_TACRED_Experiments_Log')
  except Exception as e:
    print(e)
  
  sample_list_of_relation = relation_list_extractor(RE_TACRED_dataset_40samples)
  gold_samples = gold_samples_extractor(RE_TACRED_dataset_40samples)
  Num_samples = 2

  if prompt_type == 'zero_shot':
      target_prompt = zero_shot_prompt
  elif prompt_type == 'few_shot':
      target_prompt = few_shot_prompt
  elif  prompt_type == 'CoT':
      target_prompt = CoT_prompt

  output, reasoning_list = TE_prompt_runner(target_prompt, RE_TACRED_dataset_40samples, sample_list_of_relation, Num_samples = Num_samples, CoT = prompt_type , model = model, API = API, sub_model=sub_model )    
  processed_output = TE_output_processing(output)
  nb_labels, nb_predictions =  nonbinary_evaluation(gold_samples, processed_output, orderless_relations = [])
  results = TE_metric_calculation(nb_labels, nb_predictions, sample_list_of_relation)
  print_output(output, RE_TACRED_dataset_40samples, gold_samples, nb_predictions, cot_reasoning_list = reasoning_list)
  
  if len(args) == 4:
    TE_write_output(output, RE_TACRED_dataset_40samples, 'RE_TACRED', gold_samples, nb_predictions,  prompt_type, target_prompt, results, model, cot_reasoning_list=reasoning_list)
    
  else:
    TE_write_output(output, RE_TACRED_dataset_40samples, 'RE_TACRED', gold_samples, nb_predictions,  prompt_type, target_prompt, results, sub_model, cot_reasoning_list=reasoning_list)
    