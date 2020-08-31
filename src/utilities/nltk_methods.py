# Module imports
# Please note that wordnet, punkt, and averaged_perceptron_tagger must all be downloaded
from nltk.corpus import wordnet 
import nltk

def list_of_syns_from_pos_pair(word_pos_pair: tuple) -> list:
    """Method for getting a list of synonyms to replace a word, given 
        a tuple containing a word and its verbose POS tag.  Method made
        specifically for use in conjunction with elements in the list produced
        by string_to_pos_tagged_list().

        Input:
        - word_pos_pair (tuple): tuple (word, POS) containing a word and its verbose pos tag,
            such as ('throw', 'VB')

        Output:
        - list of the word's synonyms that match its part of speech; 
            NOTE: may return an empty list"""
    # Initizalize an empty list to add good words to later
    word_syn_list = list()
    # Get the word POS from the tuple
    word_pos = get_tag_from_pos_tuple(word_pos_pair)
    condensed_tag = condense_pos_tags(word_pos)
    # Get the actual word from the tuple
    word = get_word_from_pos_tuple(word_pos_pair)

    # Now loop through synsets from wordnet and filter as you go to only get
    # useful words for replacement
    syn_list = wordnet.synsets(word)
    for syn in syn_list:
        new_word = syn.name()
        # Unpack the synonym name, which is word.POS_tag.word_version
        try:
            new_word, new_word_pos, word_version = new_word.split('.')
        except:
            continue
        # Now check to make sure the new word is actually useful for the given word
        # The condensed tags from the given word and its synonym don't match, so skip it
        if new_word_pos is not condensed_tag:
            continue
        # Now make sure the word is unique, since we don't want duplicates
        if new_word not in word_syn_list:
            word_syn_list.append(new_word)
    return word_syn_list




def list_of_syns(word: str) -> list:
    """Method for getting a list of word's synonyms from wordnet.  
    NOTE: list_of_syns_from_pos_pair() is a more valuable method, as that works
    with contextually tagged word-pos pairs, while this simply works with standalone words.

    Input:
    - word (string): the word whose synonyms are desired

    Output:
    - list of word's synonyms, where each synonym is a string; 
        NOTE: may return an empty list"""
    # Make an empty list to keep track of syns later
    string_syn_list = list()
    # Get the word part of speech to compare synonym parts of speech to
    word_pos = get_word_pos(word)
    # Use wordnet to get the synonym sets of word, then loop through them and add
    # synonyms to string_syn_list

    syn_list = wordnet.synsets(word)
    for syn in syn_list:
        new_word = syn.name()
        # Make sure the part of speech tags of the new_word is good for replacement
        new_word_pos = new_word.split('.')[1]
        verdict_tup = is_part_of_speech_acceptable_for_replacement(new_word)
        # If the word isn't good for replacement, don't add it to the synonym list;
        # similiarly, if the synonym POS isn't the same as word POs, skip it
        if verdict_tup[1] is not True or new_word_pos is not verdict_tup[0]:
            continue
        # Make sure the addition is unique, we don't need duplicates
        if new_word not in string_syn_list:
            string_syn_list.append(new_word)
    return string_syn_list


def string_to_pos_tagged_list(sentence: str) -> list:
    """Method for turning a string/sentence into a list of tuples that are (word, part_of_speech)

    Input:
    - sentence (string): the sentence that you want to POS tag

    Output:
    - list of tuples that contain word, part of speech pairs"""
    exploded_sentence = nltk.word_tokenize(sentence) # Requires punkt downloaded
    pos_tagged_list = nltk.pos_tag(exploded_sentence) # Requires averaged_perceptron_tagger downloaded
    return pos_tagged_list

def put_string_together_from_pos_tagged_list(tag_list: list) -> str:
    """Takes a list of (string, POS-tag) tuples and attempts to put the words back together into a sentence in order.
    There is an attempt to keep the punctuation marks, but no promises on how good it is.

    Input:
    - tag_list (list of tuples): the list with tuples that has been exploded and tagged

    Output:
    - a string that's the words from the list put back together"""
    # List of punctuation to check for
    punctuation_list = [',', '.', '!', '?', '-']
    # Make an empty string to append stuff to
    return_string = ""
    # Now loop through the list and put text into return_string
    for pair in tag_list:
        string = get_word_from_pos_tuple(pair)
        # If it's punctuation, plop it down without a space prefix
        # Additionally, if the first character of the string is ', then
        # we probably don't want a space in front of it either
        if string in punctuation_list or string[0] is "'":
            return_string = return_string + string
        # The string isn't punctuation, so let's assume it's a word that needs a space
        else:
            return_string = return_string + " " + string
    # Trim any leading/trailing whitespace
    return_string = return_string.strip()
    return return_string

def get_tag_from_pos_tuple(pair: tuple) -> str:
    """Returns the part of speech tag from the (word, part_of_speech) tuple.

    Input:
    - pair (tuple): tuple containing the word and the part of speech tag associated with it

    Output:
    - string that is the part of speech tag"""
    return pair[1]

def get_word_from_pos_tuple(pair: tuple) -> str:
    """Returns the word from the (word, part_of_speech) tuple.

    Input:
    - pair (tuple): tuple containing the word and the part of speech tag associated with it

    Output:
    - string that is the word"""
    return pair[0]


def is_part_of_speech_acceptable_for_replacement(word: str) -> tuple:
    """Method for determining if a word's part of speech is acceptable for synonym replacement.

    Input:
    - word (string): the word to check

    Output: 
    - tuple (bool, str): the boolean evaluates whether the word can be replaced, 
        and if it can str is the part of speech to match"""
    # Initialize variables with fail-state values
    tag = ""
    boolean = False
    word_pos = get_word_pos(word)
    # Is it a noun?
    if word_pos in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
        tag = "n"
        boolean = True
    # Is it a verb?
    elif word_pos in ["VB", "VBD", "VBG", "VBN", "VBD", "VBZ"]:
        tag = "v"
        boolean = True
    # Is it an adjective?
    elif word_pos in ["JJ", "JJR", "JJS"]:
        tag = "a"
        boolean = True
    # Is it an adverb? 
    elif word_pos in ["RB", "RBR", "RBS", "WRB"]:
        tag = "r"
        boolean = True
    return (tag, boolean)
    # https://www.guru99.com/wordnet-nltk.html
    # https://www.nltk.org/howto/wordnet.html
    # https://www.nltk.org/book/ch05.html
    # https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/

def condense_pos_tags(tag: str) -> str:
    """Condenses the verbose part of speech tags into the smaller set.  
    For example: NN | NNS | NNP | NNPS -> n

    Input:
    - tag (string): the verbose POS tag

    Output:
    - string that's the condensed POS tag"""
    # Initialize with a dummy answer
    condensed_tag = "NA"
    # Is it a noun?
    if tag in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
        condensed_tag = "n"
    # Is it a verb?
    elif tag in ["VB", "VBD", "VBG", "VBN", "VBD", "VBZ"]:
        condensed_tag = "v"
    # Is it an adjective?
    elif tag in ["JJ", "JJR", "JJS"]:
        condensed_tag = "a"
    # Is it an adverb? 
    elif tag in ["RB", "RBR", "RBS", "WRB"]:
        condensed_tag = "r"
    return condensed_tag


def get_word_pos(word: str) -> str:
    """Method that returns the part of speech tag of the word.

    Inputs:
    - word (string): the word from which to get the POS tag

    Outputs:
    - string that's the POS tag of word"""
    list_of_words_and_tags = string_to_pos_tagged_list(word)
    pair = list_of_words_and_tags[0]
    tag = pair[1]
    return tag

def clean_sentence(sentence: str) -> str:
    """Cleans the sentence by calling replace_underscore() and uppercase_first_character() on sentence.

    Inputs:
    - sentence (string): the sentence to be cleaned."""
    sentence = replace_underscores(sentence)
    sentence = uppercase_first_character(sentence)
    return sentence

def replace_underscores(sentence: str) -> str:
    """Replace underscores in a string with spaces.

    Inputs:
    - sentence (string): the string in which to replace "_" with " "

    Outputs:
    - string with '_' changed to ' '"""
    s = sentence.replace("_", " ")
    return s

def uppercase_first_character(sentence: str) -> str:
    """Uppercase the first character in the sentence.

    Inputs:
    - sentence (string) -> string in which that you want to uppercase the first character

    Outputs:
    - string that's the same as sentence but with the first character uppercased"""
    # Trim whitespace just to be safe
    sentence = sentence.strip()
    # Grab the first character and uppercase it
    char = sentence[0].upper()
    # Now put the strings back together
    nice_sentence = char + sentence[1:]
    return nice_sentence
    