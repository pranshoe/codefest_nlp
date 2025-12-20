text = """OMG!!! I luv NLP <3. It's sooooo coool! #AI #MachineLearning.
BTW, NLP isn't easy at all... but it's FUN!!! 😄😄
In college, students read blogs, tweets, research papers, emails, and chat messages daily!!!
Some texts are clean; others are FULL of typos, slangggg, emojis 😂😂, and random CAPS.
In 2025, AI-powered systems process MILLIONS of documents @ scale—fast & continuously.
If your data isn't clean and normalized properly, your model WILL fail badly!!! 💥💥
Trust me: garbage in = garbage out.""".lower()

import re

text = re.sub(r'[^\w\s<=]', '', text)

from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]


from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stopwords]


vectors_preview = {}
for word in tokens:
    if word in vectors_preview:
        vectors_preview[word]+=1
    else:
        vectors_preview[word]=1

print(vectors_preview)
