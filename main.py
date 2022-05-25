import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Bidirectional
import string
import torchtext.vocab
from sklearn.preprocessing import MinMaxScaler
from itertools import product
##############################################################################
EMBEDDING_DIM = 300
number_word_generate = 100
max_seq_size = 4
##############################################################################

##############################################################################
def pre_processing(path):
    def lyrics_corpus(text, column):
        """
        Create corpus
        :param text:
        :param column:
        :return:
        """
        try:
            text[column] = text[column].str.replace('[{}]'.format(string.punctuation), '')
            text[column] = text[column].str.replace(' +', ' ')
        except Exception as e_lyrics_corpus:
            print(e_lyrics_corpus)

        text[column] = text[column].str.lower()
        corpus = text[column]
        for item in range(len(corpus)):
            corpus[item] = corpus[item].rstrip()
            corpus[item] = corpus[item].lstrip()
        corpus = [item for item in corpus if item != '']
        return corpus

    def tokenize_corpus(corpus):
        """
        Creates the vocabulary index based on word frequency.
        So lower integer means more frequent word
        :param corpus:
        :return:
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
        return tokenizer

    dataframe = pd.read_csv(path, header=None, sep="\n")

    dataframe = dataframe.iloc[:, 0].str.rstrip(r'&, ').str.extract(r'([^,]+),([^,]+),(.+)')
    dataframe.columns = ['artist', 'title', 'lyrics']
    dataframe['lyrics'] = dataframe['lyrics'].str.replace("&", "newLine")

    dataframe = dataframe.drop_duplicates()



    # remove ["string"]
    dataframe['lyrics'] = dataframe['lyrics'].str.replace(r"\[.*\]", "")
    dataframe = dataframe[dataframe['lyrics'] != '']  # todo: find a real fix. for now we just remove the fked up lyrics

    #  developing only on the first 100 songs, remove this line when done
    # dataframe = dataframe[:5]

    dataframe, dic_midi_ans = read_midi(dataframe)
    print(f"shape dataframe - {dataframe.shape}")
    print(f"number midi file - {len(dic_midi_ans)}")
    dataframe.index = range(0, dataframe.shape[0])

    dic_midi_ans = normalize_midi_data(dic_midi_ans)

    corpus = lyrics_corpus(dataframe, 'lyrics')
    tokenizer = tokenize_corpus(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    seq = []
    for item in corpus:
        # Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and
        # replaces it with its corresponding integer value from the word_index dictionary.
        seq_list = tokenizer.texts_to_sequences([item])[0]
        song_name = dataframe.loc[dataframe['lyrics'] == item, 'title'].iloc[0]
        artist_name = dataframe.loc[dataframe['lyrics'] == item, 'artist'].iloc[0]
        # print(song_name)
        # print(artist_name)
        for i in range(1, len(seq_list)):
            n_gram = seq_list[:i + 1]
            n_gram.append(artist_name)
            n_gram.append(song_name)
            seq.append(n_gram)
    # max_seq_size = max([len(s) for s in seq])

    artists_and_songnames = list(map(lambda arr: arr[-2:], seq))
    lyric_numbers = list(map(lambda arr: arr[:-2], seq))
    seq = np.array(pad_sequences(lyric_numbers, maxlen=max_seq_size, padding='pre'))
    input_sequences, labels = seq[:, :-1], seq[:, -1]

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    glove = torchtext.vocab.GloVe(name="6B", dim=EMBEDDING_DIM)
    for word, i in tokenizer.word_index.items():
        embedding_vector = glove[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = [np.random.uniform(-4, 3) for i in range(EMBEDDING_DIM)]

    one_hot_labels = to_categorical(labels, num_classes=vocab_size)

    embedded_input_sequences = []
    embedded_input_sequences_without_midi = []
    midi_for_embedded_input_sequences = []
    for i_s in range(input_sequences.shape[0]):
        embedded_input_sequences.append([])
        embedded_input_sequences_without_midi.append([])
        midi_for_embedded_input_sequences.append([])
        artist = artists_and_songnames[i_s][0]
        songname = artists_and_songnames[i_s][1]
        artist_and_songname = artist.replace(" ", "_") + '_-_' + songname.replace(" ", "_") + '.mid'
        midi_vector_for_seq = dic_midi_ans[artist_and_songname]
        for w in range(input_sequences.shape[1]):
            word_embedding = embedding_matrix[input_sequences[i_s][w]]
            embedded_input_sequences_without_midi[i_s].append(word_embedding)
            enhanced_word_embedding = np.concatenate([word_embedding, midi_vector_for_seq])
            embedded_input_sequences[i_s].append(enhanced_word_embedding)
            midi_for_embedded_input_sequences[i_s].append(midi_vector_for_seq)

    narr_embedded_input_sequences = np.array(embedded_input_sequences)
    narr_shape = narr_embedded_input_sequences.shape
    print(f"Input shape - {narr_shape}")

    # return vocab_size, max_seq_size, input_sequences, one_hot_labels, tokenizer, embedding_matrix
    return vocab_size, max_seq_size, narr_embedded_input_sequences, one_hot_labels, tokenizer, embedding_matrix

def normalize_midi_data(dic_midi_ans):
    # normalized the dictionary of the midi data
    n_l = []
    for k in dic_midi_ans.keys():
        # print(k)
        n_l.append(dic_midi_ans[k])
    n_a = np.array(n_l)
    scaler = MinMaxScaler()
    scaler.fit(n_a)
    n_a_trnsformed = scaler.transform(n_a)
    ii = 0
    for k in dic_midi_ans.keys():
        # print(k)
        dic_midi_ans[k] = n_a_trnsformed[ii]
        ii = ii + 1

    return dic_midi_ans


def read_midi(dataframe):
    def extract_data_midi(pm):
        number_instruments = len(pm.instruments)

        # extracting the length of time signature changes
        tsc = len(pm.time_signature_changes)
        # extracting the highest probability tempo estimation
        best_tempo = pm.estimate_tempo()

        # extracting the number of notes per instrument (sum, average, min, max)
        sum_notes = 0
        for instrument in pm.instruments:
            sum_notes += len(instrument.notes)
        average_notes = sum_notes / number_instruments
        max_notes = float('-inf')
        for instrument in pm.instruments:
            max_notes = max(max_notes, len(instrument.notes))
        min_notes = float('inf')
        for instrument in pm.instruments:
            min_notes = min(min_notes, len(instrument.notes))

        # to add the average noe pitch
        max_notes_instrument = pm.instruments[0]
        mn = len(max_notes_instrument.notes)
        for instrument in pm.instruments:
            if len(instrument.notes) > mn:
                max_notes_instrument = instrument
                mn = len(instrument.notes)
        pitches = list(map(lambda note: note.pitch, max_notes_instrument.notes))
        avg_note_pitch = np.average(pitches)

        # extracting which instruments participate in the midi
        instruments_list = np.zeros(128)
        for instrument in pm.instruments:
            instrument_name = instrument.name
            try:
                instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
                instruments_list[instrument_program] = 1
            except Exception as e_ins:
                # print(e_ins)
                # print(f"extract_data_midi: name file = {name_file}")
                # if name_file not in bad_filenames:
                #     bad_filenames.append(name_file)
                pass

        midi_data_vector = np.concatenate((np.array([tsc, best_tempo, sum_notes, average_notes, max_notes, min_notes, avg_note_pitch]),instruments_list))
        return midi_data_vector

    # { index_of_dataframe : pretty_midi }
    dic_midi_ans = {}
    bad_filenames = []
    for index, row in dataframe.iterrows():
        author = row[0].replace(" ", "_")
        name_song = row[1].replace(" ", "_")
        name_file = author + "_-_" + name_song + ".mid"
        try:
            pm = pretty_midi.PrettyMIDI("midi_files/" + name_file)
            # dic_midi_ans[index] = pm
            # TODO update list of value
            midi_data_vector = extract_data_midi(pm)
            if name_file in dic_midi_ans.keys():
                print(f'duplicate!!!!!! MA ZEEEEEEEEEEEEEEE, song = {name_file}')
            dic_midi_ans[name_file] = midi_data_vector
        except Exception as e:
            # TODO need check how handle the file throw error
            print(e)
            print(f"read_midi: name file = {name_file}")
            bad_filenames.append(row[1])
            pass
    dataframe = dataframe[dataframe['title'].isin(bad_filenames) == False]

    len(dic_midi_ans)
    return dataframe, dic_midi_ans

def plot_graph(H):
    f, (ax1, ax2) = plt.subplots(2, 1)
    plt.style.use("ggplot")
    plt.figure()

    ax1.plot(H.history["accuracy"], label="train_acc")
    ax1.plot(H.history["val_accuracy"], label="val_acc")
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.legend(loc="lower left")

    ax2.plot(H.history["loss"], label="train_loss")
    ax2.plot(H.history["val_loss"], label="val_loss")
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.legend(loc="lower left")

    plt.show()


def get_model(vocab_size, input_sequences):
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    model.build(input_shape=input_sequences.shape)
    return model


##############################################################################

##############################################################################
print('preprocessing')
vocab_size, max_seq_size, input_sequences, one_hot_labels, tokenizer, embedding_matrix = pre_processing(
    "lyrics_train_set.csv")

print(f"vocab size - {vocab_size}")
print(f"max sequence size - {max_seq_size}")
print(f"input sequences shape - {input_sequences.shape}")
print(f"label shape - {one_hot_labels.shape}")
print(f"embedding matrix shape - {embedding_matrix.shape}")
##############################################################################


##############################################################################
print('getting model')
model = get_model(vocab_size, input_sequences)
model.summary()


##############################################################################


##############################################################################
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


callback_lr_decrease = tf.keras.callbacks.LearningRateScheduler(scheduler)
##############################################################################


##############################################################################
# x_train, X_val, y_train, y_val = train_test_split(input_sequences,one_hot_labels, test_size=0.2, random_state=42)

callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')

history = model.fit(
    input_sequences, one_hot_labels,
    epochs=1,
    verbose=1,
    batch_size=16,
    validation_split=0.2,
    callbacks=[callback_early_stop, callback_lr_decrease])

# history = model.fit(input_sequences, one_hot_labels, epochs=5, verbose=1, batch_size=64)
##############################################################################


##############################################################################
print('plotting graph')
plot_graph(history)
##############################################################################


##############################################################################

test_set_dataframe = pd.read_csv('lyrics_test_set.csv', header=None, sep="\n")

test_set_dataframe = test_set_dataframe.iloc[:, 0].str.rstrip(r'&, ').str.extract(r'([^,]+),([^,]+),(.+)')
test_set_dataframe.columns = ['artist', 'title', 'lyrics']
test_set_dataframe['lyrics'] = test_set_dataframe['lyrics'].str.replace("&", "newLine")
test_set_dataframe = test_set_dataframe.drop_duplicates()
test_set_dataframe['lyrics'] = test_set_dataframe['lyrics'].str.replace(r"\[.*\]", "")
test_set_dataframe = test_set_dataframe[test_set_dataframe['lyrics'] != '']  # todo: find a real fix. for now we just remove the fked up lyrics
test_set_dataframe['title'] = test_set_dataframe['title'].str[1:]

test_set_dataframe, test_set_dic_midi_ans = read_midi(test_set_dataframe)

test_set_dic_midi_ans = normalize_midi_data(test_set_dic_midi_ans)



# seed = "im"
# seed = test_set_dataframe.iloc[0,2].split()[0]
# seed_melody_parts = list(test_set_dataframe.iloc[0,:2].values)
# seed_melody_filename = seed_melody_parts[0].replace(" ", "_") + '_-_' + seed_melody_parts[1].replace(" ", "_") + '.mid'
# seed_melody = test_set_dic_midi_ans[seed_melody_filename]
seeds = ["im", "if", "i", "all", "love"]
for seed, seed_melody_key in list(product(seeds, test_set_dic_midi_ans)):
    seed_melody = test_set_dic_midi_ans[seed_melody_key]
    init_word = seed
    for _ in range(number_word_generate):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_size - 1, padding='pre')
        embedded_token_list = []
        for w in range(token_list.shape[1]):
            word_embedding = embedding_matrix[token_list[0][w]]
            enhanced_word_embedding = np.concatenate([word_embedding, seed_melody])
            embedded_token_list.append(enhanced_word_embedding)
        narr_embedded_token_list = np.array([embedded_token_list])
        predicted_probs = model.predict(narr_embedded_token_list)[0]
        predicted = np.random.choice([x for x in range(len(predicted_probs))], p=predicted_probs)
        output = ""
        seed += " " + tokenizer.index_word[predicted]
        # for word, index in tokenizer.word_index.items():
        #     if index == predicted:
        #         output = word
        #         break
        # if len(seed.split(' ')) % 5 == 0:
        #     seed += "\n " + output
        # else:
        #     seed += " " + output
    print(f" melody {seed_melody_key}, Init word '{init_word}'")
    seed = seed.replace("newline", "\n")
    print(seed)


# seed = "im"
# next_words = 100
# for _ in range(next_words):
#     token_list = tokenizer.texts_to_sequences([seed])[0]
#     token_list = pad_sequences([token_list], maxlen=max_seq_size - 1, padding='pre')
#     predicted_probs = model.predict(token_list)[0]
#     predicted = np.random.choice([x for x in range(len(predicted_probs))], p=predicted_probs)
#     output = ""
#     for word, index in tokenizer.word_index.items():
#         if index == predicted:
#             output = word
#             break
#     if len(seed.split(' ')) % 5 == 0:
#         seed += "\n " + output
#     else:
#         seed += " " + output
##############################################################################


##############################################################################

##############################################################################
