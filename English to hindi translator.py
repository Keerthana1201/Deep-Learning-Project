import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import tensorflow as tf
data = pd.read_csv("English_Hindi_Clean_New.csv", encoding='utf-8')
all_eng_words = set()
for eng in data['English']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)
all_hin_words = set()
for hin in data['Hindi']:
    for word in hin.split():
        if word not in all_hin_words:
            all_hin_words.add(word)
data['len_eng_sen'] = data['English'].apply(lambda x: len(x.split(" ")))
data['len_hin_sen'] = data['Hindi'].apply(lambda x: len(x.split(" ")))
data = data[data['len_eng_sen'] <= 20]
data = data[data['len_hin_sen'] <= 20]
max_len_src = max(data['len_hin_sen'])
max_len_tar = max(data['len_eng_sen'])
inp_words = sorted(list(all_eng_words))
tar_words = sorted(list(all_hin_words))
num_enc_toks = len(all_eng_words)
num_dec_toks = len(all_hin_words) + 1 # for zero padding
inp_tok_idx = dict((word, i + 1) for i, word in enumerate(inp_words))
tar_tok_idx = dict((word, i + 1) for i, word in enumerate(tar_words))
rev_inp_char_idx = dict((i, word) for word, i in inp_tok_idx.items())
rev_tar_char_idx = dict((i, word) for word, i in tar_tok_idx.items())
X, y = data['English'], data['Hindi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Increase batch size
batch_size = 256
def generate_batch(X=X_train, y=y_train, batch_size=batch_size):
    while True:
        for j in range(0, len(X), batch_size):
            enc_inp_data = np.zeros((batch_size, max_len_src), dtype='float32')
            dec_inp_data = np.zeros((batch_size, max_len_tar), dtype='float32')
            dec_tar_data = np.zeros((batch_size, max_len_tar, num_dec_toks), dtype='float32')
            for i, (inp_text, tar_text) in enumerate(zip(X[j:j + batch_size], y[j:j + batch_size])):
                for t, word in enumerate(inp_text.split()):
                    # Ensure the word is in the dictionary before accessing its index
                    if word in inp_tok_idx:
                        enc_inp_data[i, t] = inp_tok_idx[word]
                for t, word in enumerate(tar_text.split()):
                    # Ensure the word is in the dictionary before accessing its index
                    if word in tar_tok_idx:
                        if t < len(tar_text.split()) - 1:
                            dec_inp_data[i, t] = tar_tok_idx[word]
                        if t > 0:
                            dec_tar_data[i, t - 1, tar_tok_idx[word]] = 1.0
            yield (tf.constant(enc_inp_data), tf.constant(dec_inp_data)), tf.constant(dec_tar_data)
latent_dim = 250
enc_inps = Input(shape=(None,))
enc_emb = Embedding(num_enc_toks + 1, latent_dim)(enc_inps) # Add 1 for padding
enc_lstm = LSTM(latent_dim, return_state=True)
enc_outputs, st_h, st_c = enc_lstm(enc_emb)
enc_states = [st_h, st_c]
dec_inps = Input(shape=(None,))
dec_emb_layer = Embedding(num_dec_toks, latent_dim)
dec_emb = dec_emb_layer(dec_inps)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
dec_dense = Dense(num_dec_toks, activation='softmax')
dec_outputs = dec_dense(dec_outputs)
model = Model([enc_inps, dec_inps], dec_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy') # Use Adam optimizer for faster convergence
train_samples = len(X_train)
val_samples = len(X_test)
model.fit(x=generate_batch(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=train_samples // batch_size,
          epochs=30,
          validation_data=generate_batch(X_test, y_test, batch_size=batch_size),
          validation_steps=val_samples // batch_size)
enc_model = Model(enc_inps, enc_states)

dec_st_inp_h = Input(shape=(latent_dim,))
dec_st_inp_c = Input(shape=(latent_dim,))
dec_states_inps = [dec_st_inp_h, dec_st_inp_c]
dec_emb2= dec_emb_layer(dec_inps) # Get the embeddings of the decoder sequence
dec_outputs2, st_h2, st_c2 = dec_lstm(dec_emb2, initial_state=dec_states_inps)
dec_states2 = [st_h2, st_c2]
dec_outputs2 = dec_dense(dec_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary
dec_model = Model(
    [dec_inps] + dec_states_inps,
    [dec_outputs2] + dec_states2)
def translate(inp_seq):
    states_value = enc_model.predict(inp_seq)
    tar_seq = np.zeros((1,1))
    tar_seq[0, 0] = tar_tok_idx['START_']
    stop_cond = False
    dec_sen = ''
    while not stop_cond:
        output_toks, h, c = dec_model.predict([tar_seq] + states_value)
        sampled_tok_idx = np.argmax(output_toks[0, -1, :])
        if sampled_tok_idx in rev_tar_char_idx:
            sampled_char = rev_tar_char_idx[sampled_tok_idx]
        else:
            sampled_char = '' # or handle unknown token
        dec_sen += ' '+sampled_char
        if (sampled_char == '_END' or
           len(dec_sen) > 50):
            stop_cond = True
        tar_seq = np.zeros((1,1))
        tar_seq[0, 0] = sampled_tok_idx
        states_value = [h, c]
return dec_sen
train_gen = generate_batch(X_train, y_train, batch_size = 1)
k=0
(inp_seq, actual_output), _ = next(train_gen)
hin_sen = translate(inp_seq)
print(f'''Input English sentence: {X_train[k:k+1].values[0]}\n
          Predicted Hindi Translation: {hin_sen[:-4]}\n
          Actual Hindi Translation: {y_train[k:k+1].values[0][6:-4]}''')

