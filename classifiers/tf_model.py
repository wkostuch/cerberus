import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
# Local imports
import tfhub_embedding
import file_augmentation

#https://towardsdatascience.com/https-medium-com-piercarlo-slavazza-what-is-the-best-method-for-automatic-text-classification-a01d4dfadd
#https://www.tensorflow.org/guide/keras/sequential_model
#https://www.tensorflow.org/api_docs/python/tf/keras/layers

"""TENSOR CONVERSIONS"""

def item_to_tensor(item):
    """Converts the given item to a tensor and returns it."""
    return tf.convert_to_tensor(item)

def label_to_tensor(label: str): 
    """Converts the label to a one-hot tensor."""
    label_to_num_dict = {
        "Facts": 0,
        "Issue": 1,
        "Rule/Law/Holding": 2,
        "Analysis": 3,
        "Conclusion": 4,
        "Others": 5,
    }
    one_hot_label = to_categorical(label_to_num_dict.get(label), num_classes=6, dtype='int32')
    return one_hot_label



    """Collections of layers"""
    # 1
        #tf.keras.layers.AveragePooling1D(pool_size=4, padding='same'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Conv1D(256, kernel_size=(24), padding='same', activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        # ~25% validation accuracy
    # 2


"""BUILDING AND TRAINING THE MODEL"""


def build_model():
    """Builds the TensorFlow model and returns it.  Note: Does not train the model."""
    print("Please wait, building the model . . .")
    # Build the model layer by layer
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None,512,)),
        tf.keras.layers.AveragePooling1D(pool_size=4, padding='same'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(256, kernel_size=(24), padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, input_shape=(1024,), activation='softmax') #CHANGED THIS FOR BINARY CLASSIFICATION
    ])
    # Print the model architecture
    model.summary()
    # Compile the model with an optimizer
    # Adam default: learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.2, beta_2=0.25, epsilon=1e-07, amsgrad=False),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def train_model(train_data: list, train_labels: list, epoch_num: int, model):
    """Trains the model on the given data.
    See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    Inputs:
    - train_data (list): the list of training data as tensors,
    - train_labels (list): the list of training labels (as tensors) corresponding to train_data,
    - epoch_num (int): number of epochs to train the model on,
    - model (TensorFlow model): the model to be trained,

    Outputs:
    - the trained model"""
    # Train the model
    print("Please wait, training the model . . .")
    model.fit(train_data, train_labels, epochs=epoch_num,
                verbose=2, validation_split=0.2)
    return model
    





def read_file_and_convert_to_model_data(file_path: str) -> tuple:
    """Reads the specified CSV file and converts the sentences to 512-dimensional vectors.  
    Returns a tuple of the form (list of sentence-as-vector, list of labels)."""
    # Read the file and turn it into a list
    sentence_label_pair_list = file_augmentation.get_sentences_from_csv_file(file_path)
    # Now embed the sentences as vectors
    vector_label_pair_list = tfhub_embedding.embed_sentence_list(sentence_label_pair_list)
    # Make the two lists to keep track of things
    vector_tensor_list = list() 
    label_tensor_list = list() 
    # Dictionary for classifying labels as numbers
    """CHANGED THIS FOR BINARY CLASSIFICATION"""
    label_to_num_dict = {
        "Facts": 0,
        "Issue": 1,
        "Rule/Law/Holding": 2,
        "Analysis": 3,
        "Conclusion": 4,
        "Others": 5,
    }

    # Now split the list of tuples into a tuple of lists
    for index,pair in enumerate(vector_label_pair_list):
        vector = pair[0]
        label = pair[1]
        # Turn the vector into a tensor
        vector_tensor = item_to_tensor(vector)
        # Get a number classification for the label
        label_tensor = label_to_num_dict.get(label)
        # Add them to their respective lists
        vector_tensor_list.append(vector_tensor[0]) # The vector tensor lives inside of a redundant list, so take it out of that
        label_tensor_list.append(label_tensor)
    # Now package the lists into a tuple after converting them to Numpy arrays and return it
    # Additionally, add a "fake" dimension so that the #dims = 3 for specific TensorFlow layers
    vector_tensor_list = np.expand_dims(np.array(vector_tensor_list), axis=1)
    label_tensor_list = np.expand_dims(np.array(label_tensor_list), axis=1)
    return (vector_tensor_list, label_tensor_list)


def predict_sentence_labels(list_of_sentences: list, model) -> list:
    """Given a list_of_sentences, use the model to predict their class.

    Inputs:
    - list_of_sentences (list of strings): list of sentences to classify,
    - model (TF model): the TensorFlow model to use for predicting the sentence classes,

    Outputs:
    - list of tuples, where each tuple is (sentence, predicted_label)"""
    # Turn the list of sentences into vector embeddings
    list_of_sentence_vectors = tfhub_embedding.embed_sentence_list_no_labels(list_of_sentences)
    # Now turn those into embeddings into a format that TensorFlow models like
    vector_tensor_list = list()
    for index, vector in enumerate(list_of_sentence_vectors):
        # Turn the vector into a tensor
        vector_tensor = item_to_tensor(vector)
        # Pull out the useful part of the tensor
        vector_tensor_list.append(vector_tensor[0])
    numpy_vector_list = np.expand_dims(np.array(vector_tensor_list), axis=1)
    # Actually get the predictions from the model
    prediction_list = model.predict(numpy_vector_list)
    pair_list = list()
    # Bundle the sentences and their labels together in tuples
    for index, sentence in enumerate(list_of_sentences):
        # Due to the addition of the fake dimension, we need to pull it out at the end by accessing the 0th element
        predicted_label = convert_probabilities_to_label(prediction_list[index][0])
        pair = (sentence, predicted_label)
        pair_list.append(pair)
    return pair_list


def convert_probabilities_to_label(probability_array) -> str:
    """Converts a Numpy array of probabilities to a string label.

    Inputs:
    - probability_array (Numpy array): array of the form:
         array([9.9473441e-01, 1.5539663e-05, 2.9359373e-05, 9.5655363e-05,
         1.2352992e-03, 3.8897409e-03], dtype=float32))]
    
    Outputs:
    - string that is the label for the sentence"""
    # Dictionary for label conversion
    """CHANGED THIS FOR BINARY CLASSIFICATION"""
    label_dict = {
        0: "Facts",
        1: "Issue", #"Issue",
        2: "Rule/Law/Holding",
        3: "Analysis",
        4: "Conclusion",
        5: "Others"
    }
    # Loop through the array to find the position where the maximum value is
    max_index = 0
    max_value = 0
    for index,val in enumerate(probability_array):
        if probability_array[index] > max_value:
            max_value = probability_array[index]
            max_index = index
    # Convert the index of the maximum value to a string label
    label = label_dict.get(max_index)
    return label


def neural_net_from_file(train_data_path: str, predict_data_path: str, num_epochs=20) -> list:
    """Uses the data found at train_data_path to classify the data found at predict_data_path with a neural network.  
        Returns a list of (sentence, predicted-label) tuples."""
    # Read the data and process it into a good format
    data_pair = read_file_and_convert_to_model_data(train_data_path)
    training_data, training_labels = data_pair
    # Build and train the neural network
    model = build_model()
    trained_model = train_model(training_data, training_labels, num_epochs, model)
    # Get the sentences that we want to classify and do so
    list_of_sentences = file_augmentation.get_sentences_from_csv_file(predict_data_path, sentences_only=True)
    predicted_sentence_list = predict_sentence_labels(list_of_sentences, trained_model)
    return predicted_sentence_list
    

def evaluate_neural_network(train_file_path: str, validate_file_path: str) -> list:
    """Runs the neural network and evaluates how well it classifies the sentences."""
    # Pull the sentences from the file and build the model
    list_of_sentence_label_pairs = file_augmentation.get_sentences_from_csv_file(validate_file_path)
    list_of_sentence_predicted_label_pairs = neural_net_from_file(train_file_path, validate_file_path, num_epochs=20)
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

if __name__ == "__main__":
    
    train = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_4\team_1\s4_t1_best_labels.csv' 
    predict = r'C:\Users\wkost\VSCode\ArgumentMining\reu.unt.edu\data\AnnotationResults\reu_argument-mining_dataset\stage_3\s3_best_labels.csv'
    evaluate_neural_network(train, predict)


# Use this for data splicing https://www.tensorflow.org/guide/data ???
# https://www.tensorflow.org/tutorials/keras/text_classification Good guide for basic text classification