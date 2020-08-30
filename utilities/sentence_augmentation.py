# Module imports
import random
import nltk_methods



class SentenceAugmentor():
    """Class for augmenting sentences-as-text.
    
    Constructor takes:
    - word_deletion_rate (float): a value between 0 and 1 that determines what percentage 
        of the words in the sentence are deleted; defaults to 0
    - variation_rate (int): an integer denoting how many times to change the sentence before moving to an already-changed
        sentence for the next modification.  A higher value will correspond to slower changes in the sentence; defaults to 5
    - max_changes (int): a number to cap the changes for a particular sentence.  Also functions
        as an upper-bound on how many new sentences will be produced via the augmentation; defaults to 50
    """

    # Fields; see class doc-string for info on them
    sentence: str
    label: str
    word_deletion_rate: float
    augmented_sentence_list: list
    variation_rate: int
    max_changes: int


    def __init__(self, variation_rate=5, max_changes=50):
        """
            Constructor takes:
    - variation_rate (int): an integer denoting how many times to change the sentence before moving to an already-changed
        sentence for the next modification.  A higher value will correspond to slower changes in the sentence; defaults to 5
    - max_changes (int): a number to cap the changes for a particular sentence.  Also functions
        as an upper-bound on how many new sentences will be produced via the augmentation; defaults to 50
        """
        # Assign fields
        self.variation_rate = variation_rate
        self.max_changes = max_changes
        self.augmented_sentence_list = []

    def update_sentence_and_label_fields(self, new_sentence: str, new_label: str):
        """Updates the SentenceAugmentor's sentence and label fields to the respective inputs.

        Inputs:
        - new_sentence (string): the new sentence to be considered
        - new_label (string): the label associated with the new sentence
        
        No outputs."""
        self.sentence = new_sentence
        self.label = new_label

    def return_augmented_sentences(self) -> list:
        """Returns the augmented_sentence_list field, which is a list of strings produced 
            by augmenting the sentence (found in sentence field)

        No inputs.

        Output:
        - list of strings that are the augmented sentences"""
        return self.augmented_sentence_list

    def clear_augmented_sentences(self):
        """Sets the augmented_sentence_list field to an empty list."""
        self.augmented_sentence_list = list()

    def __synonym_replacement(self, tokenized_sentence: list) -> str:
        """Takes the tokenized sentence and replaces a random word with a random synonym.

        Inputs:
        - tokenized_sentence (list): list of (word, POS tag) tuples,
        - sentence_length (int): length of the tokenized_sentence,

        Outputs:
        - new sentence as a string.  NOTE: may return empty string"""
        sentence_length = len(tokenized_sentence)
        # Initialize the return string
        new_sentence = ""
        # Some variables to keep track of changes and attempted changes
        has_changed = False
        attempts = 0
        # Keep trying to make a change until either:
        #   1) You've made a change, OR
        #   2) You've tried to make a change for half the words in the sentence with no success
        while has_changed is not True and attempts <= sentence_length/2:
            # Grab a random word from the tokenized sentence
            index_to_change = random.randint(0, sentence_length-1)
            pair_to_change = tokenized_sentence[index_to_change]
            # Get the list of synonyms based off of that (word, POS) pair from the tokenized sentence
            list_of_syns = nltk_methods.list_of_syns_from_pos_pair(pair_to_change)
            # ...but what if it's a word that doesn't have any synonyms matching the POS tag?  
            if len(list_of_syns) < 1: 
                # Failed synonym swap, so bump up the attempts tracker by one
                attempts += 1
                continue
            # Else, the word does have synonyms we can swap the word for
            else:
                # Randomly pick a word from the synonym list
                random_pick = random.randint(0, len(list_of_syns)-1)
                new_word = list_of_syns[random_pick]
                new_word_pair = (new_word, "NA") # "NA" is a dummy POS tag
                # Now update the tokenized sentence with the new word
                tokenized_sentence[index_to_change] = new_word_pair
                # Pull the sentence back together
                new_sentence = nltk_methods.put_string_together_from_pos_tagged_list(tokenized_sentence)
                # Now let's clean up our brand new sentence really quickly
                new_sentence = nltk_methods.clean_sentence(new_sentence)
                # BUT WAIT, what if this is a duplicate?  We don't want that!
                if new_sentence in self.return_augmented_sentences():
                    # Bump up the attempts and skip this sentence
                    attempts += 1
                    continue
                # Update the flags
                has_changed = True
        return new_sentence


    def __delete_random_word(self, tokenized_sentence: list) -> str:
        """Deletes a random word from the sentence by deleted a random tuple from the tokenized_sentence.

        Inputs:
        - tokenized_sentence (list): list of (word, POS tag) tuples,
        - sentence_length (int): length of the tokenized_sentence,

        Outputs:
        - new sentence as a string.  NOTE: may return empty string"""
        sentence_length = len(tokenized_sentence)
        # Initialize return variable
        new_sentence = ""
        # Some variables to keep track of changes and attempted changes
        has_changed = False
        attempts = 0
        # Keep trying to make a change until either:
        #   1) You've made a change, OR
        #   2) You've tried to make a change for half the words in the sentence with no success
        while has_changed is not True and attempts <= sentence_length/2:
            # Grab a random word from the tokenized sentence
            index_to_change = random.randint(0, sentence_length-1)
            # Pop the word and modify the sentence length to reflect the change
            tokenized_sentence.pop(index_to_change)
            sentence_length -=1
            # Now put this new sentence back together and clean it up
            new_sentence = nltk_methods.put_string_together_from_pos_tagged_list(tokenized_sentence)
            # Now let's clean up our brand new sentence really quickly
            new_sentence = nltk_methods.clean_sentence(new_sentence)
            # BUT WAIT, what if this is a duplicate?  We don't want that!
            if new_sentence in self.return_augmented_sentences():
                # Bump up the attempts and skip this sentence
                attempts += 1
                continue
            # Update the flags
            has_changed = True
        return new_sentence


    def __swap_two_random_words(self, tokenized_sentence: list) -> str:
        """Swaps two of the words in the sentence randomly.

        Inputs:
        - tokenized_sentence (list): list of (word, POS tag) tuples,
        - sentence_length (int): length of the tokenized_sentence,

        Outputs:
        - new sentence as a string.  NOTE: may return empty string."""
        sentence_length = len(tokenized_sentence)
        # Initialize return variable to keep Pylance happy
        new_sentence = ""
        # Some variables to keep track of changes and attempted changes
        has_changed = False
        attempts = 0
        # Keep trying to make a change until either:
        #   1) You've made a change, OR
        #   2) You've tried to make a change for half the words in the sentence with no success
        while has_changed is not True and attempts <= sentence_length/2:
            # Grab a random word from the tokenized sentence
            index_to_change = random.randint(0, sentence_length-1)
            # Now grab a different random word
            counter = 0
            other_index = random.randint(0, sentence_length-1)
            # Ensure that you're not just swapping the same word with itself
            while other_index is index_to_change and counter <= sentence_length:
                other_index = random.randint(0, sentence_length-1)
                if other_index is not index_to_change:
                    break
                counter += 1
            # Now pull the old swaparoo 
            tokenized_sentence[index_to_change], tokenized_sentence[other_index] = tokenized_sentence[other_index], tokenized_sentence[index_to_change]
            # Now put this new sentence back together and clean it up
            new_sentence = nltk_methods.put_string_together_from_pos_tagged_list(tokenized_sentence)
            # Now let's clean up our brand new sentence really quickly
            new_sentence = nltk_methods.clean_sentence(new_sentence)
            # BUT WAIT, what if this is a duplicate?  We don't want that!
            if new_sentence in self.return_augmented_sentences():
                # Bump up the attempts and skip this sentence
                attempts += 1
                continue
            # Update the flags
            has_changed = True
        return new_sentence


    def __insert_random_synonym(self, tokenized_sentence: list) -> str:
        """Takes the tokenized sentence and insert's a random word's synonym into a random spot.

        Inputs:
        - tokenized_sentence (list): list of (word, POS tag) tuples,
        - sentence_length (int): length of the tokenized_sentence,

        Outputs:
        - new sentence as a string.  NOTE: may return empty string"""
        sentence_length = len(tokenized_sentence)
        # Initialize the return string
        new_sentence = ""
        # Some variables to keep track of changes and attempted changes
        has_changed = False
        attempts = 0
        # Keep trying to make a change until either:
        #   1) You've made a change, OR
        #   2) You've tried to make a change for half the words in the sentence with no success
        while has_changed is not True and attempts <= sentence_length/2:
            # Grab a random word from the tokenized sentence
            index_to_get_word_from = random.randint(0, sentence_length-1)
            pair_to_get_word_from = tokenized_sentence[index_to_get_word_from]
            # Get the list of synonyms based off of that (word, POS) pair from the tokenized sentence
            list_of_syns = nltk_methods.list_of_syns_from_pos_pair(pair_to_get_word_from)
            # ...but what if it's a word that doesn't have any synonyms matching the POS tag?  
            if len(list_of_syns) < 1: 
                # Failed synonym swap, so bump up the attempts tracker by one
                attempts += 1
                continue
            # Else, the word does have synonyms we can swap the word for
            else:
                # Randomly pick a word from the synonym list
                random_pick = random.randint(0, len(list_of_syns)-1)
                new_word = list_of_syns[random_pick]
                new_word_pair = (new_word, "NA") # "NA" is a dummy POS tag
                # Now randomly find a spot to put the new word
                index_to_place_new_word = random.randint(0, sentence_length-1)
                # Now update the tokenized sentence with the new word
                tokenized_sentence.insert(index_to_place_new_word, new_word_pair)
                sentence_length += 1
                # Pull the sentence back together
                new_sentence = nltk_methods.put_string_together_from_pos_tagged_list(tokenized_sentence)
                # Now let's clean up our brand new sentence really quickly
                new_sentence = nltk_methods.clean_sentence(new_sentence)
                # BUT WAIT, what if this is a duplicate?  We don't want that!
                if new_sentence in self.return_augmented_sentences():
                    # Bump up the attempts and skip this sentence
                    attempts += 1
                    continue
                # Update the flags
                has_changed = True
        return new_sentence


    def augment_sentence(self):
        """Creates new sentences based off of sentence field, adds those new sentences
            to the augmented_sentence_list field.

            No inputs.

            No outputs."""
        # Initialize a tracker to ensure we don't make more than the desired number of changes
        changes = 0
        # Make a queue for later
        queue = [self.sentence]

        # While we haven't made too many changes and we still have stuff to change, do work!
        while changes < self.max_changes and len(queue) > 0:
            # Take a sentence from the queue and blast it apart into a POS-tagged list
            current_sentence = queue.pop(0)
            tokenized_sentence = nltk_methods.string_to_pos_tagged_list(current_sentence)
            sentence_length = len(tokenized_sentence)
            # Now modify it according to the variation rate
            for i in range(self.variation_rate):
                # Set variable for tracking a change
                has_changed = False
                attempts = 0
                # Keep trying to make a change until either:
                #   1) You've made a change, OR
                #   2) You've tried to make a change for half the words in the sentence with no success
                while has_changed is not True and attempts <= sentence_length/2:
                    syn_sent = tokenized_sentence
                    swap_sent = tokenized_sentence
                    insert_sent = tokenized_sentence
                    del_sent = tokenized_sentence
                    successful_changes = 0
                    # Hand the sentence off to the specific augmentation methods
                    # Note that these methods can all return empty strings, so make sure to handle that
                    synonym_replaced_sentence = self.__synonym_replacement(syn_sent)
                    if synonym_replaced_sentence is not "":
                        queue.append(synonym_replaced_sentence)
                        self.augmented_sentence_list.append(synonym_replaced_sentence)
                        successful_changes += 1

                    swapped_sentence = self.__swap_two_random_words(swap_sent)
                    if swapped_sentence is not "":
                        queue.append(swapped_sentence)
                        self.augmented_sentence_list.append(swapped_sentence)
                        successful_changes += 1

                    inserted_sentence = self.__insert_random_synonym(insert_sent)
                    if inserted_sentence is not "":
                        queue.append(inserted_sentence)
                        self.augmented_sentence_list.append(inserted_sentence)
                        successful_changes +=1

                    # We don't want to delete the sentence into oblivion, so have a threshold for smallest possible sentence
                    if len(del_sent) >= 15:
                        deleted_word_sentence = self.__delete_random_word(del_sent)
                        if deleted_word_sentence is not "":
                            queue.append(deleted_word_sentence)
                            self.augmented_sentence_list.append(deleted_word_sentence)
                            successful_changes += 1
                    
                    # Now update the while loop flags
                    if successful_changes >= 4:
                        has_changed = True
                    attempts += 1
                changes += 2
                    
                    
                    




if __name__ == "__main__":
    # Short demo code
    aug = SentenceAugmentor(10,100)
    aug.update_sentence_and_label_fields("This sentence needs to have some words replaced.", "Fact")
    aug.augment_sentence()
    sentences = aug.return_augmented_sentences()
    for s in sentences:
        print(s)
    print(len(sentences))
