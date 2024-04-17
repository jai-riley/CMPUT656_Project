import sys
from triple_extraction_utils import *

API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/375121881c6ec963b21944159717e15e/ai/run/"

zero_shot_prompt = '''
The list of predicates: {}
What Subject-Predicate-Object triples are included in the following sentence? Please return the possible answers according to the list above. Require the answer only in the form: [subject, predicate, object]
The given sentence is : {}
Triples: '''


few_shot_prompt = '''
Extract Subject-Predicate-Object triples from the given sentence.

Subject:
Identify the main entity or entities in the sentence that is/are the primary focus of the relationship described by the predicate. This can be a single word or a multi-word phrase.

Predicate:
Determine the specific semantic relationship between the subject and the object based on the context of the sentence. The predicate should accurately reflect one of the following relationships predicted by your dataset:
list of Predicate: {}.
COMPARE: Denotes a comparison between the subject and object.
CONJUNCTION: Indicates that the subject and object are connected or combined.
PART-OF: Specifies that the subject is a component or part of the object.
USED-FOR: Describes the purpose or function of the subject in relation to the object.
FEATURE-OF: Highlights a characteristic or attribute of the subject belonging to the object.
HYPONYM-OF: Indicates that the subject is a more specific term or subtype of the object.
EVALUATE-FOR: Suggests an assessment or evaluation of the subject with respect to the object.
Object:
Identify the entity or entities that are associated with the subject based on the chosen predicate. The object can represent the entity receiving the comparison, conjunction, part, purpose, feature, or evaluation described by the predicate. This can be a single word or a multi-word phrase indicating the recipient, target, or characteristic of the relationship.
Ensure the correct order of subject, predicate, and object is maintained in your extracted triples.

Submit your response in the following format: Triples: [subject, predicate, object]
If there are multiple possible triples in a sentence, prioritize and return them based on the contextual relevance.

Below are some examples to have a better understanding of the task.
the given sentence: Unlike existing interest point detectors, which measure pixel-wise differences in image intensity, our detectors incorporate histogram-based representations and thus can find image regions that present a distinct distribution in the neighborhood.
Triples:  [pixel-wise differences in image intensity, EVALUATE-FOR, interest point detectors]


the given sentence: 'We then use the predicates of such clauses to create a set of domain independent features to annotate an input dataset , and run two different machine learning algorithms : SLIPPER , a rule-based learning algorithm , and TiMBL , a memory-based system.
Triples:  [rule-based learning algorithm, COMPARE, memory-based system]


the given sentence: However , when the object becomes partially or totally occluded , such local tracking is prone to failure , especially when common prediction techniques like the Kalman filter do not provide a good estimate of object parameters in future frames.
Triples:  [Kalman filter, PART-OF, prediction techniques]


the given sentence: With the aid of a logic-based grammar formalism called extraposition grammars , Chat-80 translates English questions into the Prolog subset of logic.
Triples:  [extraposition grammars, HYPONYM-OF, logic-based grammar formalism]


The given sentence: Experimental results on our dataset and the public G3D dataset both demonstrate very promising performance of our scheme.
Triples: [dataset, CONJUNCTION, G3D dataset]


The given sentence:  We show that various features based on the structure of email-threads can be used to improve upon lexical similarity of discourse segments for question-answer pairing.
Triples: [lexical similarity , FEATURE-OF , discourse segments]


The given sentence: We propose a novel probabilistic framework for learning visual models of 3D object categories by combining appearance information and geometric constraints.
Triples: [geometric constraints, USED-FOR, probabilistic framework]


The given sentence: {}
Triples:  '''


CoT_prompt = '''

Given a sentence, identify and extract relevant triples of entities connected by a predicate from the provided list.
The format of each triple should be [head, predicate, tail], separated by commas.


Instructions:

Step (1): Entity Extraction
- Extract the entities from the given sentence.
- Group sequences of words that indicate the same role in the sentence as a single entity.
- Exclude <adjectives> from entities to focus on core nouns.

Step (2): Triple Identification
- Identify a single relevant predicate from the given list that best describes the relationship between the entities. List of predicates: {}.
- Ensure that the order of subject, predicate, and object is maintained based on the predicate's meaning.
- Consider the semantic relationship between entities when selecting a predicate.

Step (3): Triple Formation
- Construct triples in the format [head, predicate, tail].
- Prioritize the triples based on your confidence in their relevance and accuracy.
- Return up to two of the most confident triples. If there is only one confident triple, return that singular triple. Ensure that the predicates in each triple are distinct.


Example 1:
Given Sentence: "With the aid of a logic-based grammar formalism called extraposition grammars, Chat-80 translates English questions into the Prolog subset of logic."

Step (1): Entity Extraction
entities = 'extraposition grammars', 'logic-based grammar formalism', 'English questions', 'Prolog subset', 'logic'

Step (2): Triple Identification
potential_triples:
- 'extraposition grammars', 'HYPONYM-OF', 'logic-based grammar formalism'
- 'Prolog subset', 'PART-OF', 'logic'

Step (3): Triple Formation, Prioritizing and selecting the two most confident triples:
['extraposition grammars', 'HYPONYM-OF', 'logic-based grammar formalism']
['Prolog subset', 'PART-OF', 'logic']


Example 2:
Given Sentence: "We propose a novel probabilistic framework for learning visual models of 3D object categories by combining appearance information and geometric constraints."

Step (1):
Entities: We, novel probabilistic framework, learning, visual models, 3D object categories, appearance information, geometric constraints

Step (2):
Potential Triples:
- 'probabilistic framework' 'USED-FOR', 'learning visual models'
- 'geometric constraints', 'USED-FOR', 'probabilistic framework'
- '3D object categories', 'PART-OF', 'visual models'

Step (3): Triple Formation, Prioritizing and selecting the two most confident triples:
['geometric constraints', 'USED-FOR', 'probabilistic framework']
['3D object categories', 'PART-OF', 'visual models']


Example 3:
Given Sentence: "Experimental results on our dataset and the public G3D dataset both demonstrate very promising performance of our scheme."

Step (1):
Entities: Experimental results, our dataset, public G3D dataset, performance, our scheme

Step (2):
Potential Triples:
- 'our dataset', 'CONJUNCTION', 'public G3D dataset'
- 'Experimental results', 'COMPARE', 'performance'

Step (3): Triple Formation, Prioritizing and selecting the two most confident triples:
['our dataset', 'CONJUNCTION', 'public G3D dataset']
['Experimental results', 'COMPARE', 'performance']


The given sentence is : {}

'''

SciERC_dataset_40samples = [
{'id': 232, 'relation': 'COMPARE', 'tokens': 'We investigate and analyze the layers of various CNN models and extensively compare between them with the goal of discovering how the layers of distributed representations within CNNs represent object pose information and how this contradicts with object category representations .', 'h': {'name': 'this', 'pos': [34, 35]}, 't': {'name': 'object category representations', 'pos': [37, 40]}} ,
{'id': 2, 'relation': 'USED-FOR', 'tokens': 'This paper describes a method for incorporating priming into an incremental probabilistic parser .', 'h': {'name': 'priming', 'pos': [7, 8]}, 't': {'name': 'incremental probabilistic parser', 'pos': [10, 13]}} ,
{'id': 63, 'relation': 'FEATURE-OF', 'tokens': 'We extract a set of heuristic principles from a corpus-based sample and formulate them as probabilistic Horn clauses .', 'h': {'name': 'probabilistic Horn clauses', 'pos': [15, 18]}, 't': {'name': 'heuristic principles', 'pos': [5, 7]}} ,
{'id': 44, 'relation': 'EVALUATE-FOR', 'tokens': 'Experiment results on ACE corpora show that this spectral clustering based approach outperforms the other clustering methods .', 'h': {'name': 'ACE corpora', 'pos': [3, 5]}, 't': {'name': 'spectral clustering based approach', 'pos': [8, 12]}} ,
{'id': 40, 'relation': 'PART-OF', 'tokens': 'Amorph recognizes NE items in two stages : dictionary lookup and rule application .', 'h': {'name': 'dictionary lookup', 'pos': [8, 10]}, 't': {'name': 'Amorph', 'pos': [0, 1]}} ,
{'id': 19, 'relation': 'EVALUATE-FOR', 'tokens': 'Techniques for automatically training modules of a natural language generator have recently been proposed , but a fundamental concern is whether the quality of utterances produced with trainable components can compete with hand-crafted template-based or rule-based approaches .', 'h': {'name': 'utterances', 'pos': [24, 25]}, 't': {'name': 'trainable components', 'pos': [27, 29]}} ,
{'id': 5, 'relation': 'USED-FOR', 'tokens': 'Our technique is based on an improved , dynamic-programming , stereo algorithm for efficient novel-view generation .', 'h': {'name': 'dynamic-programming , stereo algorithm', 'pos': [8, 12]}, 't': {'name': 'technique', 'pos': [1, 2]}} ,
{'id': 4, 'relation': 'HYPONYM-OF', 'tokens': 'We examine the relationship between the two grammatical formalisms : Tree Adjoining Grammars and Head Grammars .', 'h': {'name': 'Head Grammars', 'pos': [14, 16]}, 't': {'name': 'grammatical formalisms', 'pos': [7, 9]}} ,
{'id': 88, 'relation': 'HYPONYM-OF', 'tokens': 'Turkish is an agglutinative language with word structures formed by productive affixations of derivational and inflectional suffixes to root words .', 'h': {'name': 'Turkish', 'pos': [0, 1]}, 't': {'name': 'agglutinative language', 'pos': [3, 5]}} ,
{'id': 22, 'relation': 'CONJUNCTION', 'tokens': 'Our method takes advantage of the different way in which word senses are lexicalised in English and Chinese , and also exploits the large amount of Chinese text available in corpora and on the Web .', 'h': {'name': 'corpora', 'pos': [30, 31]}, 't': {'name': 'Web', 'pos': [34, 35]}} ,
{'id': 34, 'relation': 'EVALUATE-FOR', 'tokens': 'An experimental evaluation of summarization quality shows a close correlation between the automatic parse-based evaluation and a manual evaluation of generated strings .', 'h': {'name': 'summarization quality', 'pos': [4, 6]}, 't': {'name': 'automatic parse-based evaluation', 'pos': [12, 15]}} ,
{'id': 0, 'relation': 'USED-FOR', 'tokens': 'We present a novel method for discovering parallel sentences in comparable , non-parallel corpora .', 'h': {'name': 'method', 'pos': [4, 5]}, 't': {'name': 'discovering parallel sentences', 'pos': [6, 9]}} ,
{'id': 71, 'relation': 'FEATURE-OF', 'tokens': 'A separation method is proposed that is nearly statistically efficient -LRB- approaching the corresponding Cramér-Rao lower bound -RRB- , if the separated signals obey the assumed model .', 'h': {'name': 'Cramér-Rao lower bound -RRB-', 'pos': [14, 18]}, 't': {'name': 'separation method', 'pos': [1, 3]}} ,
{'id': 51, 'relation': 'PART-OF', 'tokens': 'In the Object Recognition task , there exists a di-chotomy between the categorization of objects and estimating object pose , where the former necessitates a view-invariant representation , while the latter requires a representation capable of capturing pose information over different categories of objects .', 'h': {'name': 'estimating object pose', 'pos': [16, 19]}, 't': {'name': 'Object Recognition task', 'pos': [2, 5]}} ,
{'id': 79, 'relation': 'PART-OF', 'tokens': 'Full digital resolution is maintained even with low-resolution analog-to-digital conversion , owing to random statistics in the analog summation of binary products .', 'h': {'name': 'random statistics', 'pos': [13, 15]}, 't': {'name': 'analog summation of binary products', 'pos': [17, 22]}} ,
{'id': 29, 'relation': 'EVALUATE-FOR', 'tokens': 'The experimental results show that the proposed histogram-based interest point detectors perform particularly well for the tasks of matching textured scenes under blur and illumination changes , in terms of repeatability and distinctiveness .', 'h': {'name': 'repeatability', 'pos': [30, 31]}, 't': {'name': 'histogram-based interest point detectors', 'pos': [7, 11]}} ,
{'id': 38, 'relation': 'PART-OF', 'tokens': 'We analyzed eye gaze , head nods and attentional focus in the context of a direction-giving task .', 'h': {'name': 'attentional focus', 'pos': [8, 10]}, 't': {'name': 'direction-giving task', 'pos': [15, 17]}} ,
{'id': 43, 'relation': 'HYPONYM-OF', 'tokens': 'This formalism is both elementary and powerful enough to strongly simulate many grammar formalisms , such as rewriting systems , dependency grammars , TAG , HPSG and LFG .', 'h': {'name': 'dependency grammars', 'pos': [20, 22]}, 't': {'name': 'grammar formalisms', 'pos': [12, 14]}} ,
{'id': 16, 'relation': 'CONJUNCTION', 'tokens': 'Multi-view constraints associated with groups of patches are combined with a normalized representation of their appearance to guide matching and reconstruction , allowing the acquisition of true three-dimensional affine and Euclidean models from multiple images and their recognition in a single photograph taken from an arbitrary viewpoint .', 'h': {'name': 'Multi-view constraints', 'pos': [0, 2]}, 't': {'name': 'normalized representation', 'pos': [11, 13]}} ,
{'id': 192, 'relation': 'COMPARE', 'tokens': 'Our technique gives a substantial improvement in paraphrase classification accuracy over all of the other models used in the experiments .', 'h': {'name': 'technique', 'pos': [1, 2]}, 't': {'name': 'models', 'pos': [15, 16]}} ,
{'id': 33, 'relation': 'PART-OF', 'tokens': 'In this theory , discourse structure is composed of three separate but interrelated components : the structure of the sequence of utterances -LRB- called the linguistic structure -RRB- , a structure of purposes -LRB- called the intentional structure -RRB- , and the state of focus of attention -LRB- called the attentional state -RRB- .', 'h': {'name': 'intentional structure', 'pos': [36, 38]}, 't': {'name': 'components', 'pos': [13, 14]}} ,
{'id': 56, 'relation': 'FEATURE-OF', 'tokens': 'The results show that the features in terms of which we formulate our heuristic principles have significant predictive power , and that rules that closely resemble our Horn clauses can be learnt automatically from these features .', 'h': {'name': 'features', 'pos': [5, 6]}, 't': {'name': 'heuristic principles', 'pos': [13, 15]}} ,
{'id': 59, 'relation': 'HYPONYM-OF', 'tokens': 'There are four language pairs currently supported by GLOSSER : English-Bulgarian , English-Estonian , English-Hungarian and French-Dutch .', 'h': {'name': 'English-Estonian', 'pos': [12, 13]}, 't': {'name': 'language pairs', 'pos': [3, 5]}} ,
{'id': 92, 'relation': 'FEATURE-OF', 'tokens': 'Here , we leverage a logistic stick-breaking representation and recent innovations in Pólya-gamma augmentation to reformu-late the multinomial distribution in terms of latent variables with jointly Gaussian likelihoods , enabling us to take advantage of a host of Bayesian inference techniques for Gaussian models with minimal overhead .', 'h': {'name': 'minimal overhead', 'pos': [45, 47]}, 't': {'name': 'Gaussian models', 'pos': [42, 44]}} ,
{'id': 54, 'relation': 'CONJUNCTION', 'tokens': 'This paper proposes a generic mathematical formalism for the combination of various structures : strings , trees , dags , graphs , and products of them .', 'h': {'name': 'strings', 'pos': [14, 15]}, 't': {'name': 'trees', 'pos': [16, 17]}} ,
{'id': 164, 'relation': 'COMPARE', 'tokens': 'We consider the problem of computing the Kullback-Leibler distance , also called the relative entropy , between a probabilistic context-free grammar and a probabilistic finite automaton .', 'h': {'name': 'probabilistic context-free grammar', 'pos': [18, 21]}, 't': {'name': 'probabilistic finite automaton', 'pos': [23, 26]}} ,
{'id': 8, 'relation': 'CONJUNCTION', 'tokens': 'Thus , in this paper , we study the problem of robust PCA with side information , where both prior structure and features of entities are exploited for recovery .', 'h': {'name': 'prior structure', 'pos': [19, 21]}, 't': {'name': 'features of entities', 'pos': [22, 25]}} ,
{'id': 6, 'relation': 'USED-FOR', 'tokens': 'A demonstration -LRB- in UNIX -RRB- for Applied Natural Language Processing emphasizes components put to novel technical uses in intelligent computer-assisted morphological analysis -LRB- ICALL -RRB- , including disambiguated morphological analysis and lemmatized indexing for an aligned bilingual corpus of word examples .', 'h': {'name': 'lemmatized indexing', 'pos': [32, 34]}, 't': {'name': 'aligned bilingual corpus', 'pos': [36, 39]}} ,
{'id': 1, 'relation': 'USED-FOR', 'tokens': 'During normal tracking conditions when the object is visible from frame to frame , local optimization is used to track the local mode of the similarity measure in a parameter space of translation , rotation and scale .', 'h': {'name': 'parameter space of translation , rotation and scale', 'pos': [29, 37]}, 't': {'name': 'local mode of the similarity measure', 'pos': [21, 27]}} ,
{'id': 74, 'relation': 'FEATURE-OF', 'tokens': 'We introduce a method to accelerate the evaluation of object detection cascades with the help of a divide-and-conquer procedure in the space of candidate regions .', 'h': {'name': 'space of candidate regions', 'pos': [21, 25]}, 't': {'name': 'divide-and-conquer procedure', 'pos': [17, 19]}} ,
{'id': 84, 'relation': 'FEATURE-OF', 'tokens': 'Experimental results from a real telephone application on a natural number recognition task show an 50 % reduction in recognition errors with a moderate 12 % rejection rate of correct utterances and a low 1.5 % rate of false acceptance .', 'h': {'name': 'natural number recognition task', 'pos': [9, 13]}, 't': {'name': 'telephone application', 'pos': [5, 7]}} ,
{'id': 67, 'relation': 'CONJUNCTION', 'tokens': 'The distinction among these components is essential to provide an adequate explanation of such discourse phenomena as cue phrases , referring expressions , and interruptions .', 'h': {'name': 'referring expressions', 'pos': [20, 22]}, 't': {'name': 'interruptions', 'pos': [24, 25]}} ,
{'id': 7, 'relation': 'USED-FOR', 'tokens': 'We propose a multi-task end-to-end Joint Classification-Regression Recurrent Neural Network to better explore the action type and temporal localiza-tion information .', 'h': {'name': 'multi-task end-to-end Joint Classification-Regression Recurrent Neural Network', 'pos': [3, 10]}, 't': {'name': 'action type', 'pos': [14, 16]}} ,
{'id': 57, 'relation': 'HYPONYM-OF', 'tokens': 'We have implemented a restricted domain parser called Plume .', 'h': {'name': 'Plume', 'pos': [8, 9]}, 't': {'name': 'restricted domain parser', 'pos': [4, 7]}} ,
{'id': 244, 'relation': 'COMPARE', 'tokens': 'We show that the trainable sentence planner performs better than the rule-based systems and the baselines , and as well as the hand-crafted system .', 'h': {'name': 'trainable sentence planner', 'pos': [4, 7]}, 't': {'name': 'baselines', 'pos': [15, 16]}} ,
{'id': 18, 'relation': 'PART-OF', 'tokens': 'Recognition of proper nouns in Japanese text has been studied as a part of the more general problem of morphological analysis in Japanese text processing -LRB- -LSB- 1 -RSB- -LSB- 2 -RSB- -RRB- .', 'h': {'name': 'Recognition of proper nouns', 'pos': [0, 4]}, 't': {'name': 'morphological analysis', 'pos': [19, 21]}} ,
{'id': 122, 'relation': 'COMPARE', 'tokens': 'Our experiments on real data sets show that the resulting detector is more robust to the choice of training examples , and substantially improves both linear and kernel SVM when trained on 10 positive and 10 negative examples .', 'h': {'name': 'detector', 'pos': [10, 11]}, 't': {'name': 'linear and kernel SVM', 'pos': [25, 29]}} ,
{'id': 36, 'relation': 'CONJUNCTION', 'tokens': 'Furthermore , this paper presents a novel algorithm for the temporal maintenance of a background model to enhance the rendering of occlusions and reduce temporal artefacts -LRB- flicker -RRB- ; and a cost aggregation algorithm that acts directly on our three-dimensional matching cost space .', 'h': {'name': 'cost aggregation algorithm', 'pos': [32, 35]}, 't': {'name': 'algorithm', 'pos': [7, 8]}} ,
{'id': 290, 'relation': 'COMPARE', 'tokens': 'Compared to the exhaustive procedure that thus far is the state-of-the-art for cascade evaluation , the proposed method requires fewer evaluations of the classifier functions , thereby speeding up the search .', 'h': {'name': 'exhaustive procedure', 'pos': [3, 5]}, 't': {'name': 'method', 'pos': [17, 18]}} ,
{'id': 45, 'relation': 'EVALUATE-FOR', 'tokens': "In particular there are three areas of novelty : -LRB- i -RRB- we show how a photometric model of image formation can be combined with a statistical model of generic face appearance variation , learnt offline , to generalize in the presence of extreme illumination changes ; -LRB- ii -RRB- we use the smoothness of geodesically local appearance manifold structure and a robust same-identity likelihood to achieve invariance to unseen head poses ; and -LRB- iii -RRB- we introduce an accurate video sequence '' reillumination '' algorithm to achieve robustness to face motion patterns in video .", 'h': {'name': 'robustness', 'pos': [89, 90]}, 't': {'name': "video sequence '' reillumination '' algorithm", 'pos': [81, 87]}}
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
    create_directory('SciERC_Experiments_Log')
  except Exception as e:
    print(e)
  SciERC_orderless_list = ["COMPARE", "CONJUNCTION"]
  sample_list_of_relation = relation_list_extractor(SciERC_dataset_40samples)
  gold_samples = gold_samples_extractor(SciERC_dataset_40samples)
  Num_samples = 2

  if prompt_type == 'zero_shot':
      target_prompt = zero_shot_prompt
  elif prompt_type == 'few_shot':
      target_prompt = few_shot_prompt
  elif  prompt_type == 'CoT':
      target_prompt = CoT_prompt

  output, reasoning_list = TE_prompt_runner(target_prompt, SciERC_dataset_40samples, sample_list_of_relation, Num_samples = Num_samples, CoT = prompt_type , model = model, API = API, sub_model=sub_model )    
  processed_output = TE_output_processing(output)
  nb_labels, nb_predictions =  nonbinary_evaluation(gold_samples, processed_output, orderless_relations = SciERC_orderless_list)
  results = TE_metric_calculation(nb_labels, nb_predictions, sample_list_of_relation)
  print_output(output, SciERC_dataset_40samples, gold_samples, nb_predictions, cot_reasoning_list = reasoning_list)
  
  if len(args) == 4:
    TE_write_output(output, SciERC_dataset_40samples, 'SciERC', gold_samples, nb_predictions,  prompt_type, target_prompt, results, model, cot_reasoning_list=reasoning_list)
    
  else:
    TE_write_output(output, SciERC_dataset_40samples, 'SciERC', gold_samples, nb_predictions,  prompt_type, target_prompt, results, sub_model, cot_reasoning_list=reasoning_list)
    