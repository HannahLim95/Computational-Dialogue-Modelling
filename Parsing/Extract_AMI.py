import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import copy
from tqdm import tqdm
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
import copy
nltk_download('wordnet')
""" Here, we process the AMI files into data structures usable for our analysis.


"""

"""
Buildup of AMI corpus:
- AjacencyPairs - each refers to a source(a) and target(b) (not always both) dialogue act, and their types
- DialogueActs - refers to words in href. Speaker is defined by capital in filename
- CorpusResources/meetings.xml defines the speakers for each meeting. global_name is speaker id, and starts with gender

"""


class Word():
    def __init__(self):
        self.id = ''
        self.text = ''
        self.start_time = ''
        self.end_time = ''
        self.meeting_id = ''
        self.participant_letter = ''

    def __init__(self, id, text, start_time, end_time, meeting_id, participant_letter):
        self.id = id
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.meeting_id = meeting_id
        self.participant_letter = participant_letter



def unpickle_or_generate(gen_fun, pickle_path, *args):
    if not os.path.isfile(pickle_path):
        obj = gen_fun(*args)
        with open(pickle_path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        with open(pickle_path, 'rb') as file:
            obj = pickle.load(file)
    return obj

def read_string_until(string, end_before, return_rest = False):
    end_index = string.index(end_before)
    if return_rest:
        return string[:end_index], string[1+end_index:]
    else:
        return string[:end_index]

def meeting_id_and_letter(filename):
    meeting_id, rest = read_string_until(filename, '.', return_rest=True)
    participant_letter = read_string_until(rest, '.')
    return meeting_id, participant_letter


def extract_words(words_directory): ###
    # Note: store the meeting id and participant letter in this as well 
    # Meeting id example: ES2002a ( out of EN2001a.A.words.xml )
    # Participant letter example: A ( out of EN2001a.A.words.xml)
    wnl = WordNetLemmatizer()
    words = {}
    words_by_id = {}
    meeting_id = ''
    participant_letter = ''
    for subdir, dirs, files in os.walk(words_directory):
        for file in sorted(files): # Sorted to ensure files are read in alphabetic order. TODO: This is an easy fix. Could make it safer by checking if key exists
            filepath = subdir + os.sep + file

            if filepath.endswith('.xml'):
                tree = ET.parse(filepath)
                root = tree.getroot()
                
                old_meeting_id = copy.deepcopy(meeting_id)
                old_participant_letter = copy.deepcopy(participant_letter)
                meeting_id, participant_letter = meeting_id_and_letter(file)
                if meeting_id != old_meeting_id: # Apparently, ES2015b.A gets read out many steps before ES2015b.B
                    words[old_meeting_id] = words_by_id
                    words_by_id = {}
                    

                for child in root:
                    id = child.attrib['{http://nite.sourceforge.net/}id']
                    start_time = float(child.attrib.get('starttime', -1))
                    end_time = float(child.attrib.get('endtime', -1))
                    if child.tag == 'w' and child.text.isalnum():
                        text = wnl.lemmatize(child.text.lower())
                    else:
                        text = False
                    words_by_id[id] = Word(id, text, start_time, end_time, meeting_id, participant_letter)
    words[meeting_id] = words_by_id # Store the last one as well

    return words

def extract_speakers(speakerspath, meetings_path): ###
    print("Extracting speakers file..")

    tree = ET.parse(meetings_path)
    root = tree.getroot()
    speakers = {}
    # Structure: {meeting_id: participant_letter: gender}
    for meeting in tqdm(root):
        tag = meeting.attrib['observation']
        speakers_in_meeting = {} 
        for speaker in meeting:
            speaker_tag = speaker.attrib['nxt_agent']
            gender = speaker.attrib['global_name'][0].lower()
            # print(gender)
            speakers_in_meeting[speaker_tag] = gender
        speakers[tag] = speakers_in_meeting
    return speakers


def dialogue_acts_words(href, words): # Retrieves words given the href from a dialogue act
    #TODO: Check in ICSI if the in-between words not only use words from a single speaker! 
    # Words files are split between speakers, so if we just use the words in a certain range, this will fuck up
    opening_bracket_index = href.find('(')
    word_ids = href[opening_bracket_index+1:]
    first_closing_bracket_index = word_ids.find(')')

    meeting_id, participant_letter = meeting_id_and_letter(href)

    start_word_id = word_ids[:first_closing_bracket_index]
    end_word_id = word_ids[first_closing_bracket_index:]
    end_opening_bracket = end_word_id.find('(')
    if end_opening_bracket != -1:
        end_word_id = end_word_id[end_opening_bracket+1:-1]
    else:
        end_word_id = None
    
    start_word_index = list(words[meeting_id].keys()).index(start_word_id)
    if end_word_id is not None:
        end_word_index = list(words[meeting_id].keys()).index(end_word_id)+1
    else:
        end_word_index = start_word_index+1
    act_words = list(words[meeting_id].values())[start_word_index:end_word_index]

    words_list = [word.text for word in act_words if word.text != False]
    start_time = act_words[0].start_time
    end_time = act_words[-1].end_time
    counter = Counter(words_list)
    # print(counter)
    return counter, start_time, end_time

def adjacency_act_id(href):
    opening_bracket_index = href.find('(') # only brackets around the id
    id = href[opening_bracket_index+1:-1] # last character is closing bracket
    return id

def extract_dialogue_acts(acts_directory, words, speakers): ### #Note: Note: About 30 of the 170 (sub)meetings do not have dialogue act/adjacency pair annotation
    print("Extracting dialogue acts..")

    # Create a dict that contains dialogue_act_id: (Counter of words, meeting_id, participant_letter, starting_time)
    # Afterwards, we can convert meeting_id and participant_letter to f or m, by using participants.xml
    # And then create a,b pairs using adjacency_pairs files
    dialogue_acts = {}
    dialogue_acts_by_id = {}
    meeting_id = ''
    for subdir, dirs, files in tqdm(os.walk(acts_directory)):
        for file in tqdm(sorted(files)):
            filepath = subdir + os.sep + file
            if filepath.endswith('dialog-act.xml'):
                
                old_meeting_id = copy.deepcopy(meeting_id)
                meeting_id, participant_letter = meeting_id_and_letter(file)
                if meeting_id != old_meeting_id:
                    dialogue_acts[old_meeting_id] = dialogue_acts_by_id
                    dialogue_acts_by_id = {}  
                tree = ET.parse(filepath)
                root = tree.getroot()
                for child in root:
                    id = child.attrib['{http://nite.sourceforge.net/}id']
                    for grandchild in child:
                        if grandchild.tag == '{http://nite.sourceforge.net/}child': # Assume each dialogue act has only one 'child' child
                            href = grandchild.attrib.get('href', None)
                            counter, start_time, end_time = dialogue_acts_words(href, words)
                    gender = speakers[meeting_id][participant_letter]
                    dialogue_acts_by_id[id] = (counter, meeting_id, gender, start_time, end_time)
    dialogue_acts[meeting_id] = dialogue_acts_by_id # Also store the last one
    return dialogue_acts #TODO: Rerun this
   
def retrieve_between_counter(dialogue_acts, meeting_id, start_time, end_time): # TODO: Count the number of dialogue acts

    acts_in_meeting = dialogue_acts[meeting_id]

    acts_between_male = [act[0] for act in acts_in_meeting.values() 
                                if (act[3] > start_time and act[4] < end_time and act[2] == 'm')]
    acts_between_female = [act[0] for act in acts_in_meeting.values() 
                                if (act[3] > start_time and act[4] < end_time and act[2] == 'f')]

    number_of_acts_between_male = len(acts_between_male)
    number_of_acts_between_female = len(acts_between_female)



    counter_acts_between_male = sum(acts_between_male, 
                                start = Counter() )
    counter_acts_between_female = sum(acts_between_female, 
                                start = Counter() )

    return counter_acts_between_male, counter_acts_between_female, acts_between_male, \
        acts_between_female, number_of_acts_between_male, number_of_acts_between_female

def extract_adjacency_pairs(acts_directory, words, speakers, dialogue_acts): ###
    print("Extracting adjacency pairs...")
    adjacency_pairs = {}

    f_m = []
    m_m = []
    m_f = []
    f_f = []

    missing_turns = 0
    total_pairs = 0
    for subdir, dirs, files in tqdm(os.walk(acts_directory)):
        for file in tqdm(files):
            filepath = subdir + os.sep + file

            if filepath.endswith('adjacency-pairs.xml'):
                meeting_id, participant_letter = meeting_id_and_letter(file)

                tree = ET.parse(filepath)
                root = tree.getroot()
                for child in root:

                    id = child.attrib['{http://nite.sourceforge.net/}id']

                    adjacency_pair = {}
                    for grandchild in child: 
                        role = grandchild.attrib['role']
                        if role != 'source' and role != 'target':
                            continue
                        href = grandchild.attrib.get('href', None)
                        dialogue_act_id = adjacency_act_id(href)

                        tag = ('a' if role == 'source' else 'b')
                        dialogue_act = dialogue_acts[meeting_id][dialogue_act_id]
                        gender = dialogue_act[2]
                        start_time = dialogue_act[3]
                        end_time = dialogue_act[4]
                        counter = dialogue_act[0]
                        adjacency_pair[tag] = (counter, gender, start_time, end_time)

                    # Now, store all in-between words of this adjacency pair
                    total_pairs += 1
                    if not ('a' in adjacency_pair.keys() and 'b' in adjacency_pair.keys()):
                        missing_turns += 1
                        continue
                    start_time = adjacency_pair['a'][3] # end time of first utterance
                    end_time = adjacency_pair['b'][2] #start time of last utterance

                    between_counters = retrieve_between_counter(dialogue_acts, meeting_id, start_time, end_time)
                    adjacency_pair['mb'], adjacency_pair['fb'], adjacency_pair['list_mb'], adjacency_pair['list_fb'],\
                        adjacency_pair['n_mb'], adjacency_pair['n_fb'] = between_counters

                    new_pair = {'a': adjacency_pair['a'][0], 'b': adjacency_pair['b'][0], 
                        'mb': adjacency_pair['mb'], 'fb': adjacency_pair['fb'], 'n_mb': adjacency_pair['n_mb'], 
                            'n_fb': adjacency_pair['n_fb'], 'list_mb':adjacency_pair['list_mb'], 'list_fb':adjacency_pair['list_fb']}
                            # This could be done more efficiently by just extending the existing adjacency_pair dict

                    if adjacency_pair['a'][1] == 'm' and adjacency_pair['b'][1] == 'm':
                        m_m.append(new_pair)
                    elif adjacency_pair['a'][1] == 'm' and adjacency_pair['b'][1] == 'f':
                        f_m.append(new_pair)
                    elif adjacency_pair['a'][1] == 'f' and adjacency_pair['b'][1] == 'm':
                        m_f.append(new_pair)
                    elif adjacency_pair['a'][1] == 'f' and adjacency_pair['b'][1] == 'f':
                        f_f.append(new_pair)
    print("Number of missing turns: ", missing_turns, total_pairs) 
    return (f_m, m_m, m_f, f_f)

def main():
    print("Extracting ICSI..")
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_AMI_path = '/../Corpora/AMI/'
    AMI_path = this_file_path + relative_AMI_path
    parsed_words_path = AMI_path + 'parsed_words.pkl'
    parsed_speakers_path = AMI_path + 'parsed_speakers.pkl'
    parsed_acts_path = AMI_path + 'parsed_acts.pkl'
    split_path = AMI_path + 'split.pkl'
    
    words_directory = AMI_path + 'words/'
    speakerspath = AMI_path + 'corpusResources/participants.xml'
    meetingspath = AMI_path + 'corpusResources/meetings.xml'
    acts_directory = AMI_path + 'dialogueActs/'


    words = unpickle_or_generate(extract_words, parsed_words_path, words_directory)

    speakers = unpickle_or_generate(extract_speakers, parsed_speakers_path, speakerspath, meetingspath)

    dialogue_acts = unpickle_or_generate(extract_dialogue_acts, parsed_acts_path, acts_directory, words, speakers)


    (f_m, m_m, m_f, f_f) = unpickle_or_generate(extract_adjacency_pairs, split_path, acts_directory, words, speakers, dialogue_acts)

    print("Lengths: ", len(f_m), len(m_m), len(m_f), len(f_f))





if __name__ == '__main__':
    main()