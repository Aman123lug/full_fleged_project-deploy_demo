from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

a = "Talks on sale to easyJet at risk of collapse: report"
new_a = [one_hot(i, 1000) for i in a.split()]
new_a = pad_sequences(new_a, padding="post")
new_a.reshape(1,-1)