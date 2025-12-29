# DO NOT EDIT THIS CODE!
import os
import pandas as pd
import numpy as np
from collections import Counter

def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts

def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)


hamlets = pd.read_csv("hamlets.csv", index_col=0)
language, text = hamlets.iloc[0]
counted_text = count_words_fast(text)

data = pd.DataFrame({
    "word": list(counted_text.keys()),
    "count": list(counted_text.values()),
    "length": [len(word) for word in list(counted_text.keys())],
    "frequency": [ "frequent" if count > 10 else "infrequent" if count <= 10 and count > 1 else "unique" for count in list(counted_text.values())]
})
'''
Without using data.groupby()
unique = 0
for i in data['frequency']:
    if i == 'unique':
        unique += 1
print(unique)
'''
data["length"] = data["word"].apply(len)

data.loc[data["count"] > 10,  "frequency"] = "frequent"
data.loc[data["count"] <= 10, "frequency"] = "infrequent"
data.loc[data["count"] == 1,  "frequency"] = "unique"
'''
Self-check without using groupby()
sub_data = pd.DataFrame({
    "language" : [language for j in list(counted_text.values())],
    "frequency" : ["frequent" if count > 10 else "infrequent" if count <= 10 and count > 1 else "unique" for count in list(counted_text.values())],
    "mean_word_length" : [np.mean(len(count)) for count in list(counted_text.keys())],
    "num_words" : [i for i in list(counted_text.values())]
   
    
})

frequent_rows = data[data["frequency"] == "infrequent"]
print(np.mean(frequent_rows ["length"]))
'''
def summarize_text(language, text):
    counted_text = count_words_fast(text)

    data = pd.DataFrame({
        "word": list(counted_text.keys()),
        "count": list(counted_text.values())
    })
    
    data.loc[data["count"] > 10,  "frequency"] = "frequent"
    data.loc[data["count"] <= 10, "frequency"] = "infrequent"
    data.loc[data["count"] == 1,  "frequency"] = "unique"
    
    data["length"] = data["word"].apply(len)
    
    sub_data = pd.DataFrame({
        "language": language,
        "frequency": ["frequent","infrequent","unique"],
        "mean_word_length": data.groupby(by = "frequency")["length"].mean(),
        "num_words": data.groupby(by = "frequency").size()
    })
    
    return(sub_data)

grouped_data = []
for i in range(3):
    language, text = hamlets.iloc[i]
    sub_data = summarize_text(language, text)
    grouped_data.append(sub_data)
'''
Old version of pandas may not support DataFrame.append() method
colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o","infrequent": "s", "unique": "^"}
import matplotlib.pyplot as plt
for i in range(grouped_data.shape[0]):
    row = grouped_data.iloc[i]
    plt.plot(row.mean_word_length, row.num_words,
        marker=markers[row.frequency],
        color = colors[row.language],
        markersize = 10
    )

color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(
        plt.plot([], [],
        color=colors[color],
        marker="o",
        label = color, markersize = 10, linestyle="None")
    )
for marker in markers:
    marker_legend.append(
        plt.plot([], [],
        color="k",
        marker=markers[marker],
        label = marker, markersize = 10, linestyle="None")
    )
plt.legend(numpoints=1, loc = "upper left")

plt.xlabel("Mean Word Length")
plt.ylabel("Number of Words")
'''
colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o","infrequent": "s", "unique": "^"}

import matplotlib.pyplot as plt


for sub_df in grouped_data:
    for _, row in sub_df.iterrows():
        plt.plot(row["mean_word_length"], row["num_words"],
                 marker=markers[row["frequency"]],
                 color=colors[row["language"]],
                 markersize=10,
                 linestyle="None")

color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(
        plt.plot([], [],
                 color=colors[color],
                 marker="o",
                 label=color, markersize=10, linestyle="None")
    )
for marker in markers:
    marker_legend.append(
        plt.plot([], [],
                 color="k",
                 marker=markers[marker],
                 label=marker, markersize=10, linestyle="None")
    )
plt.legend(numpoints=1, loc="upper left")

plt.xlabel("Mean Word Length")
plt.ylabel("Number of Words")
plt.tight_layout()
plt.show()