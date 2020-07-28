import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import copy
from tqdm import tqdm
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
nltk_download('wordnet')
""" Here, we process the ICSI files into data structures usable for our analysis.
The following file types are in the ICSI corpus:

Dialogue acts files, e.g. Bdb001.A.dialogue-acts.xml
    - Contain a list of dialogue acts for each (part of) a meeting
    - Meeting number is indicated by Bdb001
    - Part of meeting is defined by A

    - File contains the nite:root structure
        - This contains many <dialogueact> structures
            - Each dialogue act has a nite:id - id of dialogue act
            - start time
            - end time
            - type (this is a tag corresponding to the function of the dialogue act)
            - adjacency (this is a tag corresponding to the adjacency pair this dialogue act belongs to - save as string now, and process later)
            - original type (what's this? not clear, but we will save it anyway
            - participant
            - A child structure, which refers to the words, as in "Bdb001.A.words.xml#id(Bdb001.w.701)..id(Bdb001.w.702)"
                - We should decompose this to
                - File does not need to be listed, since word indices are unique - file: Bdb001.A.words.xml (or file_id: Bdb001.A.words)
                - and word_index_start
                - word_index_end
Segment files, e.g. Bdb001.A.segs.xml
    - These files contain info on 'segments', which are periods of spoken speech. Similar to dialogue acts, except not 
    with a similarly specific annotation. 
    - File contains nite: root structure
        - contains <segment> structure, which lists only the speaker and times.
    - Do not use these files, all necessary info should be in the dialogue acts

Words files, e.g. Bdb001.A.words.xml
    - contain <nite:root> structure
        - This contains many <vocalsound>, <nonvolcalsound>, <w>, <disfmarker>, <comment> and <pause> objects
        - We only need the word objects
        - these contain as tags:
            - start time
            - end time
            - c
            - k
            - qut
            - t
        - and as content:
            - A single word, or punctuation mark
    - convert this to a dict of words, with each key being the id

Speakers file, speakers.xml

"""


class Word():
    def __init__(self):
        self.id = ''
        self.text = ''
        self.start_time = ''
        self.end_time = ''
        self.c = ''
        self.t = None
        self.k = None
        self.qut = None

    def __init__(self, id, text, start_time, end_time, c, t = None, k = None, qut = None):
        self.id = id
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.c = c
        self.t = t
        self.k = k
        self.qut = qut

class Speaker():
    def __init__(self, tag, gender, age, education):
        self.tag = tag # This is the speaker id
        self.gender = gender
        self.age = age
        self.education = education

class DialogueAct():
    # TODO: Include calling of text and participant functions in initialisation
    def __init__(self, id, meeting_id, start_time, end_time, participant_id, type_, original_type, channel, 
    comment, adjacency, start_word_id, end_word_id, words, speakers):
        self.id = id
        self.meeting_id = meeting_id
        self.start_time = start_time
        self.end_time = end_time
        self.participant_id = participant_id
        # self.participant = None
        self.type_ = type_
        self.original_type = original_type
        self.channel = channel
        self.comment = comment
        self.adjacency = adjacency
        self.start_word_id = start_word_id
        self.end_word_id = end_word_id
        # self.text = None

        self.text = self.retrieve_text(words)
        self.participant = self.retrieve_participant(speakers)
        # self.adjacency_dict = self.list_adjacency()
    
    def retrieve_text(self, words_dict):
        # retrieve index in words_dict of start_word_id
        start_word_index = list(words_dict.keys()).index(self.start_word_id)
        if self.end_word_id is not None:
            end_word_index = list(words_dict.keys()).index(self.end_word_id)+1
        else:
            end_word_index = start_word_index+1

        act_words = list(words_dict.values())[start_word_index:end_word_index]
        string = ' '.join([word.text for word in act_words if word.text])
        # self.text = string
        # print(string)
        return string
    
    def retrieve_participant(self, speakers):
        participant_id = self.participant_id
        participant = speakers[participant_id]
        # print(participant.tag)
        return participant

    def list_adjacency(self):
    # also call a function to split the adjacency tags into (1) a list with all tags, and (2) a list of the number of plusses
    # Or we make it a dict! tag: (a/b, number_of_plusses, dash_number)
    # A dash indicates a split (so multiple equal utterances in the adjacency pair, by different speakers)
    # A plus indicates another utterance with the same function, by one of the same speakers. (So if the replier utters two 
    # utterances in a row, the second gets a plus)
    # Then use these tags to create a set of all adjacency tags.
        # print(self.adjacency)
        if self.adjacency is not None:
            adjacencies = self.adjacency.split('.')
            # print(adjacencies)
            # There is also a dash - 
            adjacency_dict = {}
            for tag in adjacencies:
                # print("Tag: ", tag)
                if (tag.count('a') + tag.count('b')) != 1:
                    continue
                # Select the first a or b letter, this ends the adjacency pair tag
                letter_index = max(tag.find('a'), tag.find('b')) # Removes -1 from unfound letter
                final_tag = tag[:letter_index]
                letter = tag[letter_index]
                number = 1
                # print("final tag: ", final_tag, letter_index)
                if '-' in tag:
                    number = tag[letter_index+2]
                
                no_of_plusses = tag.count('+')

                adjacency_dict[final_tag] = (letter, number, no_of_plusses)

            return adjacency_dict

        else:
            return None

def unpickle_or_generate(gen_fun, pickle_path, *args, two_files=False):
    if two_files:
        pickle_path1 = pickle_path + 'f_m_short.pkl'
        pickle_path2 = pickle_path + 'm_m_short.pkl'
        pickle_path3 = pickle_path + 'm_f_short.pkl'
        pickle_path4 = pickle_path + 'f_f_short.pkl'
        obj = gen_fun(*args)
        print("Generated pairs.")
        if len(obj)>1:
            first_part = obj[0]
            second_part = obj[1]
            third_part = obj[2]
            fourth_part = obj[3]
            def identity(thing):
                return thing
            print("starting pickling")
            unpickle_or_generate(identity, pickle_path1, first_part)
            print("Pickled one.")
            unpickle_or_generate(identity, pickle_path2, second_part)
            unpickle_or_generate(identity, pickle_path3, third_part)
            unpickle_or_generate(identity, pickle_path4, fourth_part)
            return obj
        else:
            print("Object couldn't be split in two!")
            unpickle_or_generate(gen_fun, pickle_path, *args)
    else:
        if not os.path.isfile(pickle_path):
            obj = gen_fun(*args)
            with open(pickle_path, 'wb') as file:
                pickle.dump(obj, file)
        else:
            with open(pickle_path, 'rb') as file:
                obj = pickle.load(file)
        return obj

def extract_words(words_directory):
    wnl = WordNetLemmatizer()
    words = {}
    for subdir, dirs, files in os.walk(words_directory):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith('.xml'):
                tree = ET.parse(filepath)
                root = tree.getroot()

                for child in root:
                    if child.tag == 'w':
                        # print(child.attrib)
                        id = child.attrib['{http://nite.sourceforge.net/}id']
                        start_time = child.attrib.get('starttime', None)
                        end_time = child.attrib.get('endtime', None)
                        c = child.attrib.get('c', None)
                        t = child.attrib.get('t', None)
                        k = child.attrib.get('k', None)
                        qut = child.attrib.get('qut', None)
                        text = wnl.lemmatize(child.text.lower())
                        if not text.isalnum():
                            text = False
                        words[id] = Word(id, text, start_time, end_time, c, t = t, k = k, qut = qut)
                        
                    else:
                        id = child.attrib['{http://nite.sourceforge.net/}id']
                        start_time = child.attrib.get('starttime', None)
                        end_time = child.attrib.get('endtime', None)
                        c = child.attrib.get('c', None)
                        t = child.attrib.get('t', None)
                        k = child.attrib.get('k', None)
                        qut = child.attrib.get('qut', None)
                        text = False
                        words[id] = Word(id, text, start_time, end_time, c, t = t, k = k, qut = qut)

    return words

def extract_speakers(speakerspath):
    print("Extracting speakers file..")

    tree = ET.parse(speakerspath)
    root = tree.getroot()
    speakers = {}

    for child in root:
        tag = child.attrib['tag']
        gender = child.attrib.get('gender', None)
        for grandchild in child:
            if grandchild.tag == 'age':
                age = grandchild.text
            elif grandchild.tag == 'education':
                education = grandchild.text
        speakers[tag] = Speaker(tag, gender, age, education)
    return speakers


# class DialogueTurn():
    # this class should contain the 
    #  dialogue act id 
    #  all Word objects in this dialogue act
    #  a string of the complete turn (so combination of all Word.text)
    #  participant id
    #  and in some way a reference to all adjacency pairs. We could also extract this recursively.  l

# class Adjacencyquence():
    # An object of this class should contain, for each adjacency pair: 
    # The adjacency pair id
    # A list of utterances that are part of the same sequence of adjacency pairs. 
    # A list with ranking (no plusses is zero, 1 plus is 1, etc..)
    # original: original utterance, which elicits a response (DialogueAct object)
    # original_text: words of original utterance
    # response: replying utterance (DialogueAct object)
    # utterances: |


def extract_adjacency_pairs(acts_directory, words):
    meeting_id = ''
    meeting_dict_time = {}
    meeting_dict_text = {}
    print("Extracting dialogue acts..")
    dialogue_acts_text = {}
    adjacency_dict = {}
    dialogue_acts_time = {}
    for subdir, dirs, files in tqdm(os.walk(acts_directory)):
        for file in tqdm(files):
            filepath = subdir + os.sep + file

            if filepath.endswith('.xml'):
                old_meeting_id = copy.deepcopy(meeting_id)
                meeting_id = file[:6]
                if meeting_id != old_meeting_id:
                    if len(meeting_dict_time) > 0:
                        dialogue_acts_time[meeting_id] = meeting_dict_time
                        dialogue_acts_text[meeting_id] = meeting_dict_text
                        # print(meeting_dict_time)
                        # print(meeting_dict_text)
                    meeting_dict_time = {}
                    meeting_dict_text = {}
                tree = ET.parse(filepath)
                root = tree.getroot()
                for child in tqdm(root):

                    id = child.attrib['{http://nite.sourceforge.net/}id']
                    start_time = child.attrib.get('starttime', None)
                    adjacency = child.attrib.get('adjacency', None)
                    participant_id = child.attrib.get('participant', None)

                    for grandchild in child:
                        href = grandchild.attrib.get('href', None)
                        opening_bracket_index = href.find('(')
                        word_ids = href[opening_bracket_index+1:]
                        first_closing_bracket_index = word_ids.find(')')

                        start_word_id = word_ids[:first_closing_bracket_index]
                        end_word_id = word_ids[first_closing_bracket_index:]
                        end_opening_bracket = end_word_id.find('(')
                        if end_opening_bracket != -1:
                            end_word_id = end_word_id[end_opening_bracket+1:-1]
                        else:
                            end_word_id = None
                    start_word_index = list(words.keys()).index(start_word_id)
                    if end_word_id is not None:
                        end_word_index = list(words.keys()).index(end_word_id)+1
                    else:
                        end_word_index = start_word_index+1
                    act_words = list(words.values())[start_word_index:end_word_index]
                    list_of_words = [word.text.lower() for word in act_words if word.text] # TODO: Here, we should call preprocess for words
                    # string = ' '.join(list_of_words)
                    string = list_of_words
                    # In dialogue_acts we store the id: text
                    # In dialogue_acts_time we store id:time
                    meeting_dict_time[id] = start_time
                    meeting_dict_text[id] = (list_of_words, participant_id[0])
                    if adjacency is not None:
                        adjacencies = adjacency.split('.')
                        # retrieve index in words_dict of start_word_id
                        # TODO: COnvert this to function
                        for tag in adjacencies:
                            if (tag.count('a') + tag.count('b')) != 1:
                                continue # This filters out faulty tags (sadly these exist in the dataset)
                            # Select the first a or b letter, this ends the adjacency pair tag
                            letter_index = max(tag.find('a'), tag.find('b')) # Removes -1 from unfound letter
                            final_tag = file[:6] + tag[:letter_index] # Tags are reused in different meetings, this splits them up
                            letter = tag[letter_index]
                            number = 1
                            # print("final tag: ", final_tag, letter_index)
                            if '-' in tag:
                                number = tag[letter_index+2]
                            no_of_plusses = tag.count('+')
                            if final_tag in adjacency_dict.keys() and letter in adjacency_dict[final_tag].keys():
                                for item_index, item in enumerate(adjacency_dict[final_tag][letter]): # loop through all existing acts for the current tag
                                    if number == item[0] and participant_id == item[3]:

                                        text = item[2]
                                        if len(text) < no_of_plusses + 1:
                                            text += ['']*(no_of_plusses + 1 - len(text))
                                        text[no_of_plusses] = string
                                        insert_tupl = (number, no_of_plusses, text, participant_id, item[4])
                                        adjacency_dict[final_tag][letter][item_index] = insert_tupl
                                        break
                                else:
                                    string_list = ['']*(no_of_plusses+1)
                                    string_list[no_of_plusses] = string
                                    insert_tupl = (number, no_of_plusses, string_list, participant_id, start_time)
                                    adjacency_dict[final_tag][letter].append(insert_tupl)
                            elif final_tag in adjacency_dict.keys(): # Only the letter is not in there yet. We need to insert it

                                string_list = ['']*(no_of_plusses+1)
                                string_list[no_of_plusses] = string
                                insert_tupl = (number, no_of_plusses, string_list, participant_id, start_time)
                                adjacency_dict[final_tag][letter] = [insert_tupl]
                            else: # So final_tag is not in there yet. We need to insert it
                                string_list = ['']*(no_of_plusses+1)
                                string_list[no_of_plusses] = string
                                insert_dict = {letter: [(number, no_of_plusses, string_list, participant_id, start_time)]}
                                adjacency_dict[final_tag] = insert_dict



    return adjacency_dict, dialogue_acts_time, dialogue_acts_text

def process_adjacency_pairs(adjacency_dict):

    adjacency_pairs = []
    # Now we sould convert the 'text' entries to counters, and separate each a-b pair.
    for AP_tag in adjacency_dict.keys():
        # print(AP_tag, adjacency_dict[AP_tag])
        # this contains a list of dialogue acts that belong to this one adjacency pair
        for dialogue_act in adjacency_dict[AP_tag].get('a', []):
            a_counter = Counter()
            text = dialogue_act[2]
            # a_starttime = dialogue_act[4]
            for utterance in text:
                list_of_tokens = utterance
                a_counter.update(list_of_tokens)

            a_participant_gender = dialogue_act[3][0]
            for dialogue_act in adjacency_dict[AP_tag].get('b', []):
                # print("Found b!")
                b_counter = Counter()
                text = dialogue_act[2]
                # b_starttime = dialogue_act[4]
                for utterance in text:
                    list_of_tokens = utterance #.lower() #.split(' ')
                    b_counter.update(list_of_tokens)
                b_participant_gender = dialogue_act[3][0]
                dic = {
                    'a': 
                        {   'counter': a_counter, 
                            'gender': a_participant_gender
                        },
                    'b': 
                        {   'counter': b_counter, 
                            'gender': b_participant_gender
                        }   
                        }
                adjacency_pairs.append(dic)
    return adjacency_pairs

def process_inbetween(adjacency_dict, dialogue_acts_time, dialogue_acts_text):
    print("Processing in between")
    # adjacency_pairs = []
    f_m = []
    m_m = []
    m_f = []
    f_f = []
    # Now we sould convert the 'text' entries to counters, and separate each a-b pair.
    for AP_tag in tqdm(adjacency_dict.keys()):
        # print(AP_tag, adjacency_dict[AP_tag])
        meeting_id = AP_tag[:6]


        time_values = np.array(list(dialogue_acts_time[meeting_id].values())).astype(np.float)
        time_sorted_indices = np.argsort(time_values) # Sorts in ascending order
        text_values = list(dialogue_acts_text[meeting_id].values()) #Each value is a (text, gender) tuple, so this returns a list of those tuples
        # print(dialogue_acts_time[meeting_id].values())
        # print(time_values)
        # print(dialogue_acts_text)
        # text_sorted = list(dialogue_acts_text.values())[time_sorted_indices] 
        # print("A: ", len(time_values), len(text_values))
        time_text_dict = {time_values[index]:text_values[index] for index in time_sorted_indices}


        # this contains a list of dialogue acts that belong to this one adjacency pair
        for dialogue_act in adjacency_dict[AP_tag].get('a', []):
            a_counter = Counter()
            text = dialogue_act[2]
            a_starttime = dialogue_act[4]
            for utterance in text:
                list_of_tokens = utterance #.lower() #.split(' ')
                a_counter.update(list_of_tokens)

            a_participant_gender = dialogue_act[3][0]
            for dialogue_act in adjacency_dict[AP_tag].get('b', []):
                # print("Found b!")
                b_counter = Counter()
                text = dialogue_act[2]
                b_starttime = dialogue_act[4]
                for utterance in text:
                    list_of_tokens = utterance #.lower().split(' ')
                    b_counter.update(list_of_tokens)
                b_participant_gender = dialogue_act[3][0]

                # in_between_counter = Counter()
                # Make a male and female in between counter, which are separately counted
                number_of_male_between = 0
                number_of_female_between = 0
                male_between_counter = Counter()
                female_between_counter = Counter()
                male_between_list = []
                female_between_list = []
                in_between_list = []
                for (time, text) in time_text_dict.items():
                    if time > float(a_starttime) and time < float(b_starttime):
                        max_number_in_between = 10
                        in_between_list.append((text[0], text[1]))
                        if len(in_between_list)>max_number_in_between:
                            _ = in_between_list.pop(0)
                for text in in_between_list:
                    if text[1] == 'f':
                        this_counter = Counter(text[0])
                        female_between_counter +=  this_counter
                        number_of_female_between += 1
                        female_between_list.append(this_counter)
                    elif text[1] == 'm':
                        this_counter = Counter(text[0])
                        male_between_counter += this_counter
                        number_of_male_between += 1
                        male_between_list.append(this_counter)
                    else:
                        print("Gender not defined correctly.")



                dic = {
                    'a': 
                        {   'counter': a_counter, 
                            'gender': a_participant_gender
                            # 'time': a_starttime
                        },
                    'b': 
                        {   'counter': b_counter, 
                            'gender': b_participant_gender
                            # 'time': b_starttime
                        },
                    'male_between':
                        {
                            'counter': male_between_counter,
                            'gender': 'm',
                            'number': number_of_male_between,
                            'list': male_between_list                             
                        },
                    'female_between':
                        {
                            'counter': female_between_counter,
                            'gender': 'f',
                            'number': number_of_female_between,
                            'list': female_between_list
                        }
                        }

                new_pair = {'a': dic['a']['counter'], 'b': dic['b']['counter'], 
                    'mb': dic['male_between']['counter'], 'fb': dic['female_between']['counter'], 
                    'n_mb': dic['male_between']['number'], 'n_fb': dic['female_between']['number'],
                    'list_mb': dic['male_between']['list'], 'list_fb': dic['female_between']['list']}
                if dic['a']['gender'] == 'm' and dic['b']['gender'] == 'm':
                    m_m.append(new_pair)
                elif dic['a']['gender'] == 'm' and dic['b']['gender'] == 'f':
                    f_m.append(new_pair)
                elif dic['a']['gender'] == 'f' and dic['b']['gender'] == 'm':
                    m_f.append(new_pair)
                elif dic['a']['gender'] == 'f' and dic['b']['gender'] == 'f':
                    f_f.append(new_pair)

    # return adjacency_pairs
    return (f_m, m_m, m_f, f_f)

def split_genders(adjacency_pairs):
    f_m = []
    m_m = []
    m_f = []
    f_f = []

    for adjacency_pair in adjacency_pairs:
        new_pair = {'a': adjacency_pair['a']['counter'], 'b': adjacency_pair['b']['counter']}
        if adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'm':
            m_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'f':
            f_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'm':
            m_f.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'f':
            f_f.append(new_pair)
    
    return (f_m, m_m, m_f, f_f)

def split_genders_between(in_between_pairs):
    f_m = []
    m_m = []
    m_f = []
    f_f = []

    for adjacency_pair in in_between_pairs:
        new_pair = {'a': adjacency_pair['a']['counter'], 'b': adjacency_pair['b']['counter'], 
            'mb': adjacency_pair['male_between']['counter'], 'fb': adjacency_pair['female_between']['counter'], 
            'n_mb': adjacency_pair['male_between']['number'], 'n_fb': adjacency_pair['female_between']['number'],
            'list_mb': adjacency_pair['male_between']['list'], 'list_fb': adjacency_pair['female_between']['list']}
        if adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'm':
            m_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'f':
            f_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'm':
            m_f.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'f':
            f_f.append(new_pair)
    
    return (f_m, m_m, m_f, f_f)

def main():
    print("Extracting ICSI..")
    # TODO: Create only a single time-based dict in extract_adjacency_pairs, which has all info necessary.
    # TODO: Merge process_adjacency_pairs and process_inbetween functions
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_ICSI_path = '/../Corpora/ICSI/'
    ICSI_path = this_file_path + relative_ICSI_path
    parsed_words_path = ICSI_path + 'parsed_words.pkl'
    parsed_acts_path = ICSI_path + 'parsed_acts.pkl'
    parsed_adjacency_path = ICSI_path + 'adjacency_dict_V2.pkl'
    adjacency_pair_list_pickle = ICSI_path + 'adjacency_list.pkl'
    parsed_speakers_path = ICSI_path + 'parsed_speakers.pkl'
    parsed_gender_split = ICSI_path + 'parsed_gender_split.pkl'
    words_directory = ICSI_path + 'Words/'
    acts_directory = ICSI_path + 'DialogueActs/'
    speakerspath = ICSI_path + 'speakers.xml'
    all_dialogue_acts_path = ICSI_path + 'speakers.xml'
    in_between_path = ICSI_path + 'in_between.pkl'
    split_in_between_path = ICSI_path + 'in_between_split.pkl'
    words = unpickle_or_generate(extract_words, parsed_words_path, words_directory)

    print(len(words))

    speakers = unpickle_or_generate(extract_speakers, parsed_speakers_path, speakerspath)
    print(len(speakers))

    ## some statistics
    female_dict = {}
    for speaker in speakers:
        


    adjacency_dict, dialogue_acts_time, dialogue_acts_text = unpickle_or_generate(extract_adjacency_pairs, parsed_adjacency_path, acts_directory, words)

    # adjacency_pairs = unpickle_or_generate(process_adjacency_pairs, adjacency_pair_list_pickle, adjacency_dict)

    (f_m_b, m_m_b, m_f_b, f_f_b) = unpickle_or_generate(process_inbetween, ICSI_path, adjacency_dict, dialogue_acts_time, dialogue_acts_text, two_files=True)
    print("Finished in between pairs")
    # (f_m, m_m, m_f, f_f) = unpickle_or_generate(split_genders, parsed_gender_split, adjacency_pairs)
    #  = unpickle_or_generate(split_genders_between, split_in_between_path, in_between_pairs, two_files=True)

    # print('\n \n')
    # print(len(f_m_b[2]['fb']))
    # number_of_not_in_between =  sum( [1 for pair in f_m_b if len(pair['mb']) == 0] ) + \
    #                             sum( [1 for pair in m_m_b if len(pair['mb']) == 0] ) + \
    #                             sum( [1 for pair in m_f_b if len(pair['fb']) == 0] ) + \
    #                             sum( [1 for pair in f_f_b if len(pair['fb']) == 0] ) 
    # total_number_of_pairs = len(f_m_b)+len(m_m_b) + len(m_f_b) + len(f_f_b)
    # print('Pairs with turns in between: ', total_number_of_pairs - number_of_not_in_between,'out of', total_number_of_pairs)
    # # Each of these is a list of dicts, with four keys: a, b, and mb (male between) and fb (female between)
    

    # number_of_empty_counters =  sum( [1 for pair in f_m_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )   + \
    #                             sum( [1 for pair in m_m_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )   + \
    #                             sum( [1 for pair in m_f_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )  + \
    #                             sum( [1 for pair in f_f_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )  
    # # total_number_of_pairs = len(f_m_b)+len(m_m_b) + len(m_f_b) + len(f_f_b)
    # print("Number of empty counters: ", number_of_empty_counters, 'out of', total_number_of_pairs)
    # print(f_m_b[0])
    # print(m_m_b[0])
    # print(f_f_b[0])
    # print(m_f_b[0])
    # print(adjacency_dict)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # parser.add_argument('--save_interval', type=int, default=500,
    #                     help='save every SAVE_INTERVAL iterations')
    # parser.add_argument('--interpolate', action='store_true', 
    #                     help = 'Include this flag to plot interpolation between two generated digits.')
    main()