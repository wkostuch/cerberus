import numpy as np
#from numpy.core.numeric import tensordot
from sklearn.naive_bayes import ComplementNB

from utilities import tfhub_embedding
from utilities import file_augmentation


def split_data_pairs(list_of_vector_label_pairs: list, binary_classification=False) -> tuple:
    """Splits a list of (vector, label) pairs into a tuple of (vector_list, label_list)
        after converting the labels to numerical categories.  Returns the tuple of lists."""
    # Dictionary for label conversion, if we want binary classification then Facts = 0, everying else = 1
    if binary_classification is True:
        label_to_num_dict = {
            "Facts": 0,
            "Issue": 1,
            "Rule/Law/Holding": 1,
            "Analysis": 1,
            "Conclusion": 1,
            "Others": 1,           
        }
    else:
        label_to_num_dict = {
            "Facts": 0,
            "Issue": 1,
            "Rule/Law/Holding": 2,
            "Analysis": 3,
            "Conclusion": 4,
            "Others": 5,
        }
    # Our two lists to add stuff to
    list_of_vectors = list()
    list_of_labels = list()
    # Loops through the tuples and rebundle the data 
    print("Arranging data, please wait . . .")
    """
    
    TURN THIS INTO A GENERATOR
    ~1.03GB per 1.6k sentences
    
    """
    for index,pair in enumerate(list_of_vector_label_pairs):
        if index % 10000 is 0:
            print(f"    Arranged {index} pieces of data so far")
        # Get the vector, map all of its values to be non-negative, then bundle it in a tuple
        vector = tuple(map(lambda val: abs(val), tfhub_embedding.get_vector_from_pair(pair)))
        numerical_label = label_to_num_dict.get(tfhub_embedding.get_label_from_pair(pair))
        # Now add the data to its respective list
        list_of_vectors.append(vector)
        list_of_labels.append(numerical_label)
    print("Data arrangement comple.")
    return (list_of_vectors, list_of_labels)


def train_naive_bayes(list_of_vector_label_pairs: list, binary_classification=False, alpha=1.0, norm=False):
    """Builds and trains a Naive Bayes classifier on a list of (vector, label) tuples.
        Returns the Naive Bayes classifier."""
    # Repackage the data for training the classifier
    list_of_vector_tuples, list_of_labels = split_data_pairs(list_of_vector_label_pairs, binary_classification)
    # Make and fit the classifier
    classifier = ComplementNB(alpha=1.0, norm=True)
    print("Please wait, training the Naive Bayes classifier now. . .")
    classifier.fit(list_of_vector_tuples, list_of_labels)
    print("Training complete.")
    return classifier


def predict_sentence_vector_labels(classifier, list_of_sentence_vectors_to_predict: list, binary_classification=False) -> list:
    """Uses the Naive Bayes classifier to predict the labels for a list of sentences that have been embedded as vectors.  
        Returns a list of predicted labels."""
    # Return list to add stuff to
    list_of_predictions = list()
    # Dictionary for decoding labels, split upon binary vs non-binary classification
    if binary_classification is True:
        label_dict = {
            0: "Facts",
            1: "Non-fact"
        }
    else:    
        label_dict = {
            0: "Facts",
            1: "Issue",
            2: "Rule/Law/Holding",
            3: "Analysis",
            4: "Conclusion",
            5: "Others"
        }
    # Pull apart the vectors and map them vectors to have only positive values to be consistent with the model parameters
    list_of_only_vectors = list()
    for item in list_of_sentence_vectors_to_predict:
        list_of_only_vectors.append(list(map(lambda value: abs(value), item[0])))
    # Now predict some labels!
    predicted_labels = classifier.predict(list_of_only_vectors)
    # Loop through the list and decode back into string labels
    for numerical_label in predicted_labels:
        string_label = label_dict.get(numerical_label)
        list_of_predictions.append(string_label)
    return list_of_predictions


def predict_sentence_labels(classifier, list_of_sentences: list, binary_classification=False) -> list:
    """Predicts the labels for a list of sentences-as-strings.  Returns a list of (sentence, prediction) tuples."""
    # Embed the sentences into vectors
    vector_list = tfhub_embedding.embed_sentence_list_no_labels(list_of_sentences)
    # Predict the labels
    prediction_list = predict_sentence_vector_labels(classifier, vector_list, binary_classification)
    # Now repackage the data
    tuple_list = list()
    print("Predicting sentence labels now.")
    for index,sentence in enumerate(list_of_sentences):
        if index % 500 is 0:
            print(f"    Completed {index} predictions so far . . .")
        sentence_label_pair = (sentence, prediction_list[index])
        tuple_list.append(sentence_label_pair)
    print("Label predictions completed.")
    return tuple_list


def make_classifier_from_file_path(file_path_to_train_sentences: str, binary_classification=False, alpha=1.0, norm=False):
    """Trains a Naive Bayes classifier and returns it.

    Inputs:
    - file_path_to_train_sentences (string): the file-path to the sentences and labels to use for training,
    - binary_classification (boolean): whether the classifier should do Facts or Non-fact classifaction or for each label,

    Outputs:
    - A trained Naive Bayes classifier"""
    # Read the file with the sentences and labels
    sentences = file_augmentation.get_sentences_from_csv_file(file_path_to_train_sentences)
    # Embed those as vectors
    vector_sentences = tfhub_embedding.embed_sentence_list(sentences)
    # Now make the classifier
    classifier = train_naive_bayes(vector_sentences, binary_classification, alpha, norm)
    return classifier


def naive_bayes_from_file(file_path_to_train_sentences: str, file_path_to_test_sentences: str, binary_classification=False, alpha=1.0, norm=False) -> list:
    """Uses the sentences and labels from file_path_to_train_sentences to make a Naive Bayes classifier and then classifies
        the sentences found at file_path_to_test_sentences and returns a list of (sentence, predicted-label) pairs."""
    classifier = make_classifier_from_file_path(file_path_to_train_sentences, binary_classification, alpha, norm)
    test_sentences_with_labels = file_augmentation.get_sentences_from_csv_file(file_path_to_test_sentences)
    # Now split up the test sentences and labels
    list_of_sentences = list()
    for pair in test_sentences_with_labels:
        sentence = pair[0]
        list_of_sentences.append(sentence)
    # Now predict labels
    predictions = predict_sentence_labels(classifier, list_of_sentences, binary_classification)
    return predictions


def evaluate_naive_bayes(train_file_path: str, validate_file_path: str, binary_classification: bool, alpha=1.0, norm=False) -> list:
    """Evaluates how accurate the Naive Bayes classifier is on already classified data.  Returns a list of 
        (sentence, label, predicted-label) tuples."""
    # Dictionaries and lists that depend on whether the classifcation is binary or not
    if binary_classification is True:
        label_to_num_dict = {
            "Facts": 0,
            "Issue": 1,
            "Rule/Law/Holding": 1,
            "Analysis": 1,
            "Conclusion": 1,
            "Others": 1, 
            "Non-fact": 1,          
        }
        label_dict = {
            0: "Facts",
            1: "Non-fact"
        }
        confusion_matrix = [[0 for i in range(2)] for j in range(2)]
    else:
        label_to_num_dict = {
            "Facts": 0,
            "Issue": 1,
            "Rule/Law/Holding": 2,
            "Analysis": 3,
            "Conclusion": 4,
            "Others": 5,
        }
        label_dict = {
            0: "Facts",
            1: "Issue",
            2: "Rule/Law/Holding",
            3: "Analysis",
            4: "Conclusion",
            5: "Others"
        }
        confusion_matrix = [[0 for i in range(6)] for j in range(6)]
    # Make a classifier with the training data
    classifier = make_classifier_from_file_path(train_file_path, binary_classification, alpha, norm)
    # Get the validation sentences
    list_of_sentence_label_pairs = file_augmentation.get_sentences_from_csv_file(validate_file_path)
    list_of_sentences = list()
    for pair in list_of_sentence_label_pairs:
        sentence = pair[0]
        list_of_sentences.append(sentence)
    # Predict some labels
    list_of_sentence_predicted_label_pairs = predict_sentence_labels(classifier, list_of_sentences, binary_classification)
    return_list = list()
    # Now match predictions with accuracy, but first, make some variables for keeping track of things
    total_classified = 0 
    correct_classifications = 0
    # Now put everything back together as (sentence, label, predicted label)
    for index, pair in enumerate(list_of_sentence_predicted_label_pairs):
        sentence = pair[0]
        label = list_of_sentence_label_pairs[index][1]
        predicted_label = pair[1]
        # Package it up nicely
        sentence_label_predicted_label_tuple = (sentence, label, predicted_label)
        return_list.append(sentence_label_predicted_label_tuple)
        # Finally, update our confusion matrix and our counters for accuracy
        label_num = label_to_num_dict.get(label) 
        predicted_label_num = label_to_num_dict.get(predicted_label)
        confusion_matrix[label_num][predicted_label_num] = confusion_matrix[label_num][predicted_label_num] + 1
        # Update our counters as well
        total_classified += 1
        if label_num is predicted_label_num:
            correct_classifications +=1
    # Print the categories
    print("\n")
    for index, label_list in enumerate(confusion_matrix):
        print(label_dict.get(index), end="   ")
    # Print the overall accuracy
    accuracy = correct_classifications / total_classified
    print('\n')
    print(f"Alpha: {alpha}")
    print(f"Classification accuracy: {accuracy}")
    # Print the confusion matrix, with accuracy rates per label
    for index, label_list in enumerate(confusion_matrix):
        total_classifications_in_category = sum(label_list)
        correct_classifications = label_list[index]
        label_accuracy = correct_classifications / (total_classifications_in_category + 0.00001) # Prevent zero division error
        print(label_list, end="    ")
        print(f"Accuracy for {label_dict.get(index)}, {correct_classifications} / {total_classifications_in_category} classified correctly: {label_accuracy}")
    return return_list

if __name__ == "__main__":
    train = r'.\legal_data\stage_4\team_1\s4_t1_best_labels.csv' 
    test = r'.\legal_data\stage_3\s3_best_labels.csv'
    for i in range(1, 2):
        alpha = i
        evaluate_naive_bayes(train, test, binary_classification=False, alpha=alpha, norm=True)
        print('\n')
        print("~*~*~*~*~*~*~")
        print('\n')