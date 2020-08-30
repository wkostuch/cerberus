# Module imports
import csv 
from sentence_augmentation import SentenceAugmentor


def augment_file(file_path: str, save_when_done: bool, new_file_name: str):
    """Read a CSV file with the sentence data and run it through the SentenceAugmentor.  
    Optionally, save the old and augmented data in a new CSV file.

    Inputs:
    - file_path (string): the path to the CSV file to augment;
    - save_when_done (boolean): True writes the augmented sentences as a CSV file, False does not;
    - new_file_name (string): the name of the file to write the data to when finished.

    No outputs."""

    # Open the CSV file and read it
    with open(file_path, newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        # Loop through the rows in the CSV file
        for row_index, row_data in enumerate(csv_reader):
            # Skip the first row as that's not sentence data
            if row_index is 0:
                continue
            # Grab the data we want
            sentence = row_data[0]
            sentence_label_number = row_data[1]
            sentence_label = row_data[2]
            
            # Make an augmentor, but weight more towards non-fact sentences
            if sentence_label is "Fact":
                augmentor = SentenceAugmentor(2, 1)
                continue
            else:
                augmentor = SentenceAugmentor(2, 4)
            
            # Now start augmenting the sentences
            # First update the sentence and label fields
            augmentor.update_sentence_and_label_fields(sentence, sentence_label)
            # Actually do the augmentation and then get that list of sentences
            augmentor.augment_sentence()
            augmented_sentences_list = augmentor.return_augmented_sentences()
            # Clear the augmented sentences field in the augmentor object so it's clear for the next time
            augmentor.clear_augmented_sentences()
            # Add the original sentence to the augmented sentences list so we keep that 
            augmented_sentences_list.append(sentence)
            # Now clean up the augmented sentences for easy writing
            labeled_augmented_sentence_list = list()
            for current_sentence in augmented_sentences_list:
                # Make a small list with [sentence, label_id, label_text]
                small_list = [current_sentence, sentence_label_number, sentence_label]
                # Now add that small list to the total list
                labeled_augmented_sentence_list.append(small_list)
            # Progress report while running it
            if row_index % 100 is 0:
                print(f"Processed {row_index} sentences so far.")
           # print(labeled_augmented_sentence_list)
            # Do we want to keep this data?  
            #if save_when_done is True:
                # Now write the new sentences and their labels to a CSV file
            with open(new_file_name, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for item in labeled_augmented_sentence_list:
                    writer.writerow(item)



def get_sentences_from_csv_file(file_path: str, sentences_only=False) -> list:
    """Reads the CSV file found at file_path, returns the data as a list of (sentence, label) pairs.

    Inputs:
    - file_path (string): file path leading to CSV file with sentence data,
    - sentences_only (boolean): if True, only the sentence will be grabbed,

    Outputs:
    - if sentences_only is True: list of sentences,
    - if sentences_only is False: List of (sentence, label) tuples with both entries being strings"""
    return_list = list()
    # Open the CSV file and read it
    with open(file_path, newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        # Loop through the rows in the CSV file
        for row_index, row_data in enumerate(csv_reader):
            # Skip the first row as that's not sentence data
            if row_index is 0:
                continue
            # Grab the data we want
            sentence = row_data[0]
            sentence_label_number = row_data[1]
            sentence_label = row_data[2]
            # Package into a list or a tuple depending on the sentences_only flag setting
            if sentences_only is True:
                return_list.append(sentence)
                continue
            word_label_pair = (sentence, sentence_label)
            return_list.append(word_label_pair)
    return return_list



def augment_file_and_get_sentences(file_path: str) -> list:
    """Augments the data found in the CSV file from file_path, returns those sentences in a list.

    Inputs:
    - file_path (string): file path leading to CSV file with sentence data

    Outputs:
    - list containing the augmented data"""
    # First augment the desired file and save it
    saved_file_path = ''.join([file_path[:-4], "_augmented.csv"])
    augment_file(file_path, True, saved_file_path)
    sentence_label_pair_list = get_sentences_from_csv_file(saved_file_path)
    return sentence_label_pair_list
            

            
            
            



if __name__ == "__main__":
    augment_file(r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\no_facts\s2_and_s3_no_facts.csv', 
                    True, "s2_and_s3_no_facts_augmented.csv")

    


