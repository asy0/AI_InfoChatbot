from sentence_transformers import CrossEncoder

# Load Cross-Encoder model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(user_query, embedded_sentences, top_k=3):
    """
    First use cosine similarity to get top candidates, then rerank them with a Cross-Encoder.
    Returns a sorted list of ((sentence, _, source), score).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    bi_encoder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    query_embedding = bi_encoder.encode([user_query])[0]

    # Bi-Encoder step: Get top-N candidates by cosine similarity
    scored_candidates = []
    for sentence, embedding, source in embedded_sentences:
        score = cosine_similarity([query_embedding], [embedding])[0][0]
        scored_candidates.append(((sentence, embedding, source), score))

    # Take top 10 by bi-encoder and rerank them with cross-encoder
    top_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:10]
    cross_input = [(user_query, candidate[0]) for (candidate, _) in top_candidates]
    cross_scores = cross_encoder.predict(cross_input)

    # Combine sentences with new cross-encoder scores
    reranked = list(zip([x[0] for x in top_candidates], cross_scores))
    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:top_k]
    return reranked
