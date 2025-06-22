from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)
bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

def rerank_with_cross_encoder(user_query, embedded_sentences, top_k=40):
    print("Reranking with Cross-Encoder...")

    # Bi-Encoder encoding on GPU
    query_emb = bi_encoder.encode(f"query: {user_query}", convert_to_tensor=True, device=device)

    # Cosine similarity to pre-filter
    top_candidates = []
    for sentence, embedding, source in embedded_sentences:
        embedding = embedding.to(device)
        score = float(util.cos_sim(query_emb, embedding)[0][0])
        top_candidates.append(((sentence, embedding, source), score))

    top_candidates = sorted(top_candidates, key=lambda x: x[1], reverse=True)[:20]

    # Filter redundant candidates
    filtered = []
    seen = []
    for candidate, _ in top_candidates:
        sentence, embedding, source = candidate
        if all(float(util.cos_sim(embedding, other)[0][0]) < 0.95 for other in seen):
            seen.append(embedding)
            filtered.append(candidate)

    # Cross-encode final shortlist
    cross_inputs = [(user_query, sentence) for sentence, _, _ in filtered]
    cross_scores = cross_encoder.predict(cross_inputs)

    reranked = list(zip(filtered, cross_scores))
    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:top_k]
    return reranked
