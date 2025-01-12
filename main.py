from collections import Counter, defaultdict
import math
import re


# funkcja do tokenizacji tekstu (usuwanie kropek, przecinków i innych znaków interpunkcyjnych)
def tokenize(text):
    text = re.sub(r'[.,!?;]', '', text)  # Usuwanie znaków interpunkcyjnych
    return text.lower().split()


# funkcja do obliczania prawdopodobieństwa zapytania q dla dokumentu d
def calculate_query_likelihood(doc_tokens, query_tokens, collection_model, lambda_value=0.5):
    doc_length = len(doc_tokens)
    doc_counter = Counter(doc_tokens)

    score = 1
    print("Dokument tokenized:", doc_tokens)
    for token in query_tokens:
        p_doc = doc_counter[token] / doc_length if doc_length > 0 else 0
        p_coll = collection_model.get(token, 0)
        p_smoothed = lambda_value * p_doc + (1 - lambda_value) * p_coll

        # Mnożenie prawdopodobieństw
        score *= p_smoothed

    print(f"Final score for document: {score:.6f}\n")
    return score


# główna funkcja
def query_likelihood_ranking(n, documents, query, lambda_value=0.5):
    tokenized_docs = [tokenize(doc) for doc in documents]
    query_tokens = tokenize(query)

    # model kolekcji
    all_tokens = [token for doc in tokenized_docs for token in doc]
    collection_length = len(all_tokens)
    collection_counter = Counter(all_tokens)
    collection_model = {token: count / collection_length for token, count in collection_counter.items()}


    # obliczanie prawdopodobieństw dla każdego dokumentu
    scores = []
    for idx, doc_tokens in enumerate(tokenized_docs):
        print(f"Processing document {idx}...")
        score = calculate_query_likelihood(doc_tokens, query_tokens, collection_model, lambda_value)
        scores.append((idx, score))

    # sortowanie dokumentów
    sorted_scores = sorted(scores, key=lambda x: (-x[1], x[0]))

    # zwracanie posortowanych indeksów dokumentów
    return [idx for idx, _ in sorted_scores]


# wejście
if __name__ == "__main__":
    n = int(input().strip())
    documents = []
    print()
    for _ in range(n):
        documents.append(input().strip())

    query = input().strip()

    result = query_likelihood_ranking(n, documents, query)
    print(result)
