'''
For those that haven't kept up with their Greek mythology, Cereberus is the 3-headed dog who guards the underworld.  
Since this file combines three different ML models to classify sentences, it seemed appropriate.
'''

import file_augmentation
import tfhub_embedding
import naive_bayes
import knn_512
import tf_model

def run_cerberus(train_data_path: str, no_fact_train_data_path: str, predict_data_path: str, k=8, num_epochs=15) -> list:
    """Runs Cerberus to first split sentences on a binary prediction of Fact or Non-fact, then
        uses KNN to further classify the Non-fact sentences into their respective labels.
        
        Inputs:
        - train_data_path (string): the filepath to the data on which to train the ML models,
        - no_fact_train_data_path (string): the filepath to a set of data with no Fact sentences, used for second layer of KNN classification,
        - predict_data_path (string): the filepath to the data which you wish to classify,
        - k (int): the number of neighbors to use for KNN,
        - num_epochs (int): the number of epochs to train the neural network on,
        
        Outputs:
        - list of (sentence, predicted-label) tuples."""
    # Run Cerberus to get the binary predictions of Fact or Non-fact
    cerberus_binary_predictions = run_cerberus_binary(train_data_path, predict_data_path, k, num_epochs)
    # Now sort through that list to get a list of Fact and Non-fact sentences
    fact_sentence_prediction_pair_list = list() # List of (sentence, predicted-label) tuples
    non_fact_sentences_list = list() # List of just the sentences
    for pair in cerberus_binary_predictions:
        sentence = pair[0]
        label = pair[1]
        if label is "Facts":
            fact_sentence_prediction_pair_list.append(pair)
        else:
            non_fact_sentences_list.append(sentence)
    # Now classify the Non-fact sentences via KNN
    non_fact_sentence_prediction_pair_list = knn_512.knn_from_file(k*2, non_fact_sentences_list, no_fact_train_data_path)
    # Now merge the lists back together and return that list
    return [*fact_sentence_prediction_pair_list, *non_fact_sentence_prediction_pair_list]


def evaluate_cerberus(train_data_path: str, no_fact_train_data_path: str, predict_data_path: str, k=5, num_epochs=15) -> list:
    """Runs full Cerberus and evaluates how well it classifies the sentences."""
    # Pull the sentences from the file and also run Cerberus
    list_of_sentence_label_pairs = file_augmentation.get_sentences_from_csv_file(predict_data_path)
    list_of_sentence_predicted_label_pairs = run_cerberus(train_data_path, no_fact_train_data_path, predict_data_path, k, num_epochs)
    # Make some dictionaries for use later
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
    print(f"Classification accuracy, {correct_classifications} / {total_classified} correctly: {accuracy}")
    # Print the confusion matrix, with accuracy rates per label
    for index, label_list in enumerate(confusion_matrix):
        total_classifications_in_category = sum(label_list)
        correct_classifications = label_list[index]
        label_accuracy = correct_classifications / (total_classifications_in_category + 0.00001)
        print(label_list, end="    ")
        print(f"Accuracy for {label_dict.get(index)}, {correct_classifications} / {total_classifications_in_category} classified correctly: {label_accuracy}")
    return return_list


def run_cerberus_binary(train_data_path: str, predict_data_path: str, k=5, num_epochs=20) -> list:
    """Runs Cerberus to predict binary sentence classification.  

    Inputs:
    - train_data_path (string): filepath to the data to train the models on,
    - predict_data_path (string): filepath to the data whose classification we wish to predict,
    - k (int): the k to use for KNN,
    - num-epochs (int): the number of epochs to train the neural net on,

    Outputs:
    - A list of (sentence, predicted-label) tuples where predicted-label is the majority vote of the binary classification
        given by Naive Bayes, KNN, and a neural network."""
    # Run our three different classifiers on the given data, where each returns a list of (sentence, predicted-label) tuples
    naive_bayes_sentence_prediction_list = naive_bayes.naive_bayes_from_file(train_data_path, predict_data_path, binary_classification=True, norm=True)
    knn_sentence_prediction_list = knn_512.knn_strictly_from_file(k, predict_data_path, train_data_path)
    neural_net_sentence_prediction_list = tf_model.neural_net_from_file(train_data_path, predict_data_path, num_epochs)
    # Now sort through all of that and pick the majority-vote prediction
    sentence_final_prediction_list = list()
    for index,pair in enumerate(naive_bayes_sentence_prediction_list):
        fact_predictions = 0
        non_fact_predictions = 0
        # Pull apart our tuples
        naive_bayes_sentence, naive_bayes_prediction = pair
        knn_sentence, knn_prediction = knn_sentence_prediction_list[index]
        neural_net_sentence, neural_net_prediction = neural_net_sentence_prediction_list[index]
        # Make sure our sentences are all the same
        if naive_bayes_sentence == knn_sentence == neural_net_sentence:
            # Now check the predictions
            if naive_bayes_prediction is "Facts": fact_predictions+=1
            else: non_fact_predictions+=1

            if knn_prediction is "Facts": fact_predictions+=1
            else: non_fact_predictions+=1

            if neural_net_prediction is "Facts": fact_predictions+=1
            else: non_fact_predictions +=1

            # Now pick the label based on majority vote
            if fact_predictions > non_fact_predictions: predicted_label = "Facts"
            else: predicted_label = "Non-fact"
            # Bundle our data back up nicely
            pair = (naive_bayes_sentence, predicted_label)
            sentence_final_prediction_list.append(pair)
    return sentence_final_prediction_list
    

def evaluate_cerberus_binary(train_data_path, predict_data_path, k=5, num_epochs=20) -> list:
    """Runs Cerberus on the provided inputs and evaluates its performance.  Returns a list of
        (sentence, label, predicted-label) tuples."""
    # Get our sentences with labels attached and also run Ceberus
    list_of_sentence_label_pairs = file_augmentation.get_sentences_from_csv_file(predict_data_path)
    list_of_sentence_predicted_label_pairs = run_cerberus_binary(train_data_path, predict_data_path, k, num_epochs)
    # Make some dictionaries for use later
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
    print(f"Classification accuracy, {correct_classifications} / {total_classified} correct: {accuracy}")
    # Print the confusion matrix, with accuracy rates per label
    for index, label_list in enumerate(confusion_matrix):
        total_classifications_in_category = sum(label_list)
        correct_classifications = label_list[index]
        label_accuracy = correct_classifications / total_classifications_in_category
        print(label_list, end="    ")
        print(f"Accuracy for {label_dict.get(index)}, {correct_classifications} / {total_classifications_in_category} classified correctly: {label_accuracy}")
    return return_list


if __name__ == "__main__":
    #train_file_path = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_4\team_1\s4_t1_best_labels.csv'
    train_file_path = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_3\s3_best_labels.csv' #1

    #train_no_facts = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\no_facts\s2_and_s3_no_facts.csv' #1
    #train_no_facts = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\src\data_augmentation\s2_and_s3_no_facts_augmented.csv'
    train_no_facts = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_4\team_1\s4_t1_no_facts.csv'

    #validate_file_path = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_4\team_1\s4_t1_best_labels.csv' #1
    validate_file_path = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_3\s3_best_labels.csv'

    #full_predictions = run_cerberus(train_file_path, train_no_facts, validate_file_path, k=4, num_epochs=10)
    evaluate_cerberus(train_file_path, train_no_facts, validate_file_path, k=10, num_epochs=15)
    print(f"Train on {train_file_path}") 
    print(f"No-fact data from {train_no_facts}")
    print(f"Validation data from {validate_file_path}")


    '''
    ROUND 1:
    Train: s3 best labels
    Validate: s4 best labels
    
    Results:
    '''



    #predictions = evaluate_cerberus_binary(train_file_path, validate_file_path, k=4, num_epochs=n)
    