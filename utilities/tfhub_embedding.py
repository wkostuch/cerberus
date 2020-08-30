# TensorFlow Hub Universal Sentence Encoder: https://tfhub.dev/google/universal-sentence-encoder/4
import tensorflow
import tensorflow_hub as hub
import file_augmentation

""" Example from the TF Hub page
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print(embeddings)
"""


def embed_sentence_without_label(sentence: str, embedder):
    """Embeds a sentence as a 512-dimensional vector."""
    vector = embedder([sentence])
    return vector


def embed_sentence_list_no_labels(list_of_sentences: list) -> list:
    """Embeds each sentence in the list to a 512-dimensional vector, returns a list containing the vectors."""
    # Load the TFHub module with the sentence encoder
    print("Loading sentence embedder, please wait . . .")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    vector_list = list()
    print("Beginning sentence embedding. . .")
    for index, sentence in enumerate(list_of_sentences):
        if index % 10000 is 0:
            print(f"    Embedded {index} sentences so far")
        vector = embed_sentence_without_label(sentence, embed)
        vector_list.append(vector)
    print(f"Sentence embedding finished.")
    return vector_list


def embed_sentence_with_label(sentence_label_pair: tuple, embedder) -> tuple:
    """Takes a (sentence, label) tuple and embeds the sentence to a vector.

    Inputs:
    - sentence_label_pair (tuple): the pair to embed the sentence,
    - embedder (TFHub model): the sentence embedder

    Outputs:
    - (sentence_as_vector, label)"""
    sentence, label = sentence_label_pair
    vector = embedder([sentence])
    return (vector, label)

    
def embed_sentence_list(list_of_pairs: list) -> list:
    """Given a list of (sentence, label) tuples, return a list with the sentences as vectors.

    Inputs:
    - list_of_pairs (list): the list to use for vectorizing,

    Outputs:
    - list of (vector, label) tuples"""
    # Load the TFHub module with the sentence encoder
    print("Loading sentence embedder, please wait . . .")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    vector_list = list()
    print("Beginning sentence embedding. . .")
    for index, pair in enumerate(list_of_pairs):
        if index % 10000 is 0:
            print(f"    Embedded {index} sentences so far")
        vector_pair = embed_sentence_with_label(pair, embed)
        vector_list.append(vector_pair)
    print("Sentence embedding finished.")
    return vector_list


def get_vector_from_pair(tensor_label_pair: tuple):
    """Returns the actual tensor from the (tensor, label) pair."""
    return tensor_label_pair[0][0]

def get_label_from_pair(tensor_label_pair: tuple):
    """Returns the label from the (tensor, label) pair."""
    return tensor_label_pair[1]


    



if __name__ == "__main__":
    sentences = [("Hello my good sir", "greeting"), ("What a truly wonderful day today, isn't it?", "question")]
    vector_embeddings = embed_sentence_list(sentences)
    print(vector_embeddings[0]) # First tuple pair
    print(vector_embeddings[0][0]) # Tensor from that tuple
    print(vector_embeddings[0][0][0]) # The actual tensor where you can access each item in it
    print(vector_embeddings[0][0][0][0]) # First value in the tensor
    print(get_vector_from_pair(vector_embeddings[0]))
    print(get_label_from_pair(vector_embeddings[0]))
    
