import spacy

# Load the large English NLP model
nlp = spacy.load('en_core_web_md')

# Define the terms to compare
cat = nlp("cat")
monkey = nlp("monkey")
banana = nlp("banana")

# Compute similarity scores
cat_monkey = cat.similarity(monkey)
cat_banana = cat.similarity(banana)
monkey_banana = monkey.similarity(banana)

# Print the similarity scores
print("Similarity between cat and monkey:", cat_monkey)
print("Similarity between cat and banana:", cat_banana)
print("Similarity between monkey and banana:", monkey_banana)

# Define a new term to compare
book = nlp("book")

# Compute similarity score with cat
book_cat = book.similarity(cat)

# Print the similarity score
print("Similarity between book and cat:", book_cat)

# Load the smaller English NLP model
nlp = spacy.load('en_core_web_sm')

# Compute similarity scores with the smaller model
cat_monkey = cat.similarity(monkey)
cat_banana = cat.similarity(banana)
monkey_banana = monkey.similarity(banana)
book_cat = book.similarity(cat)

# Print the similarity scores
print("Similarity between cat and monkey:", cat_monkey)
print("Similarity between cat and banana:", cat_banana)
print("Similarity between monkey and banana:", monkey_banana)
print("Similarity between book and cat:", book_cat)
