import numpy as np
import tfhub_embedding
import file_augmentation


def distance_between_two_vectors(vec1, vec2) -> float:
    """Returns the Euclidian distance between two vectors of real numbers.
    Vector embeddings given by tfhub_embedding."""
    return np.linalg.norm(vec1-vec2)


def map_embedded_pairs_to_distance(embedded_pair_list: list, given_vector) -> list:
    """Maps the list of (vector, label) tuples to (distance, label) where distance is the distance
        between each vector and given_vector."""
    return list(map(lambda pair: (distance_between_two_vectors(given_vector, tfhub_embedding.get_vector_from_pair(pair)), pair[1]), embedded_pair_list))


def sort_list_of_pairs_by_distance(list_of_pairs):
    """Sorts the list of (distance, label) pairs by the distance value.  Sort occurs in-place."""
    list_of_pairs.sort(key=lambda pair: pair[0])


def tag_sentences_via_knn(k: int, sentence_list: list, list_of_vector_label_pairs: list) -> list:
    """Uses KNN algorithm (Euclidian distance) to tag a list of sentences.

    Inputs:
    - k (int): number of neighbors to check,
    - sentence_list (list of strings): the list of sentences to tag via KNN,
    - list_of_sentence_label_pairs (list): list of tuples of (sentence-as-vector, label-as-string) form to use as neighbors

    Outputs:
    - list of tuples of the form (sentence, predicted-label)"""
    sentences_copy = sentence_list.copy()
    # Two dictionaries for encoding/decoding labels and positions in lists
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
    # Initizalize the list to return at the end of the method
    list_of_sentence_predicted_label_pairs = list()
    # Get a list of your sentence as vectors
    vectorized_sentences = tfhub_embedding.embed_sentence_list_no_labels(sentence_list)
    # Now loop through that list and sort the list of vectors by distance
    print("Tagging sentences, please wait . . .")
    for index, vector_to_tag in enumerate(vectorized_sentences):
        if index % 500 is 0: 
            print(f"    Tagged {index} sentences so far")
        list_copy = list_of_vector_label_pairs.copy()
        list_of_distances = map_embedded_pairs_to_distance(list_copy, vector_to_tag)
        # Sort the list by distance to the sentence we want to tag
        sort_list_of_pairs_by_distance(list_of_distances)
        # Tuple to keep track of what neighbors are in what classification bucket
        neighbor_tracking = [0, 0, 0, 0, 0, 0]
        # Now loop through the k-closest neighbors and track their labels
        for i in range(k):
            # Get the pair and pull it apart to get the label
            pair = list_of_distances[i]
            label = pair[1]
            # Map the label to its corresponding entry in the tracking list
            label_num = label_to_num_dict.get(label)
            neighbor_tracking[label_num] = neighbor_tracking[label_num] + 1
        # Now find the max value and index in our tracking list
        max_value = 0
        max_index = 0
        for small_index, value in enumerate(neighbor_tracking):
            if value >= max_value: # This >= break makes it so that ties are decided in favor of non-fact labels
                max_value = value
                max_index = small_index
        # Now decode the index back to the label it represents as a string
        label = label_dict.get(max_index)
        # Bundle everything up into a nice tuple and add it to our list
        sentence_label_pair = (sentences_copy[index], label)
        list_of_sentence_predicted_label_pairs.append(sentence_label_pair)
    print("Sentence tagging completed.")
    return list_of_sentence_predicted_label_pairs


def knn_from_file(k: int, list_of_sentences: list, file_path: str,) -> list:
    """Uses KNN to classify each sentence in list_of_sentences using sentences and labels from the file at file_path, k is number of neighbors to check.  
        Returns a list of (sentence, label) tuples."""
    # Read the CSV file for target vectors
    sentence_label_pair_list = file_augmentation.get_sentences_from_csv_file(file_path)
    # Embed those sentences as the target vectors
    embedded_sentence_label_pair_list = tfhub_embedding.embed_sentence_list(sentence_label_pair_list)
    # Run KNN on the sentences with respect to the target vectors
    predicted_sentence_list = tag_sentences_via_knn(k, list_of_sentences, embedded_sentence_label_pair_list)
    return predicted_sentence_list


def knn_strictly_from_file(k: int, file_path_to_sentences: str, file_path_to_target_sentences: str) -> list:
    """Uses the sentences found at file_path_to_target_sentences to classify the sentences found at file_path_to_sentences.  
        Returns a list of (sentence, predicted-label) tuples."""
    # Read the sentences-to-predict from file and map them into the appropriate format
    list_of_sentences = file_augmentation.get_sentences_from_csv_file(file_path_to_sentences, sentences_only=True)
    # Now use the other KNN method
    return knn_from_file(k, list_of_sentences, file_path_to_target_sentences)


def evaluate_knn_with_confusion_matrix(k: int, file_path_to_validation_sentences: str, file_path_to_target_sentences: str) -> list:
    """Evaluates the KNN classifier by printing the confusion matrix for that classification attempt.  Also returns a list of 
        (sentence, label, predicted-label) tuples.  
        Validation sentences: the sentences whose class you wish to predict.
        Target sentences: the sentences you're using as neighbors to predict classifications."""
    # Make a 2-d array for the confusion matrix, 6x6 matrix filled with 0s
    confusion_matrix = [[0 for i in range(6)] for j in range(6)]
    #confusion_matrix = [[0 for i in range(2)] for j in range(2)] """CHANGED FOR BINARY"""
    # Two dictionaries for encoding/decoding labels and positions in lists
    """THIS IS WHAT WAS CHANGED FOR BINARY"""
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
    # Read the validation sentences into a list of (sentence, actual-label) pairs
    list_of_sentence_label_pairs = file_augmentation.get_sentences_from_csv_file(file_path_to_validation_sentences)
    # Now get just the sentences from that and throw them into the KNN classifier
    list_of_validation_sentences = list(map(lambda pair: pair[0], list_of_sentence_label_pairs.copy()))
    list_of_sentence_predicted_label_pairs = knn_from_file(k, list_of_validation_sentences, file_path_to_target_sentences)
    # Make a list to add things too
    return_list = list()
    # Counters for accuracy
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
    for index, label_list in enumerate(confusion_matrix):
        print(label_dict.get(index), end="   ")
    # Print the overall accuracy
    accuracy = correct_classifications / total_classified
    print('\n')
    print(f"Classification accuracy with k = {k}, {correct_classifications} / {total_classified}: {accuracy}")
    # Print the confusion matrix, with accuracy rates per label
    for index, label_list in enumerate(confusion_matrix):
        total_classifications_in_category = sum(label_list)
        correct_classifications = label_list[index]
        label_accuracy = correct_classifications / (total_classifications_in_category + 0.00001) # Prevent zero division error
        print(label_list, end="    ")
        print(f"Accuracy for {label_dict.get(index)}, {correct_classifications} / {total_classifications_in_category} classified correctly: {label_accuracy}")
    return return_list



if __name__ == "__main__":
    file_path_target = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_4\team_1\s4_t1_best_labels.csv'  
    file_path_validaton = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_3\s3_best_labels.csv'
    for k in range(10,11):
        l = evaluate_knn_with_confusion_matrix(k, file_path_validaton, file_path_target)
        print('\n')
        print("~*~*~*~*~*~*~*~*~*~")
        print('\n')


