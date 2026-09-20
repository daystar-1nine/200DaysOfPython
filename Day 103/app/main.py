"""Main orchestration pipeline for Day 103 Word2Vec Engine."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.corpus import get_corpus
from app.data.tokenizer import tokenize_corpus
from app.embeddings.word2vec import Word2Vec
from app.similarity.nearest_neighbors import compute_similarity_matrix, extract_all_nearest_neighbors
from app.evaluation.embedding_analysis import analyze_embeddings
from app.visualization.embeddings import generate_visualizations
from app.report import generate_report

def main():
    print("=== STARTING DAY 103 WORD2VEC ENGINE (FROM SCRATCH) ===")
    
    # 1. Load and tokenize corpus
    corpus = get_corpus()
    tokenized_corpus = tokenize_corpus(corpus)
    print(f"Loaded {len(corpus)} sentences, {sum(len(d) for d in tokenized_corpus)} total tokens.")
    
    # Save processed tokens
    config.DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_PROCESSED, "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f, indent=2)
        
    # 2. Train Word2Vec Model
    print("Training Word2Vec Skip-Gram Model with Negative Sampling...")
    model = Word2Vec(
        embedding_dim=config.EMBEDDING_DIM,
        window_size=config.WINDOW_SIZE,
        negative_samples=config.NEGATIVE_SAMPLES,
        learning_rate=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        seed=config.RANDOM_STATE
    )
    model.fit(tokenized_corpus)
    print(f"Training Complete! Initial Loss: {model.epoch_losses[0]:.4f} -> Final Loss: {model.epoch_losses[-1]:.4f}")
    
    # 3. Save Outputs
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, model.W_in)
    with open(config.VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(model.vocab.word2idx, f, indent=2)
        
    loss_df = pd.DataFrame({"epoch": range(1, len(model.epoch_losses) + 1), "loss": model.epoch_losses})
    loss_df.to_csv(config.TRAINING_LOSS_PATH, index=False)
    
    # 4. Similarity Matrix for Selected Words
    selected_words = ["cat", "dog", "kitten", "puppy", "milk", "meat", "python", "java", "code"]
    present_words = [w for w in selected_words if w in model.vocab]
    sim_df = compute_similarity_matrix(model.W_in, present_words, model.vocab)
    sim_df.to_csv(config.SIMILARITY_MATRIX_PATH)
    
    # 5. Nearest Neighbors
    neighbors_df = extract_all_nearest_neighbors(model.W_in, model.vocab, top_k=4)
    neighbors_df.to_csv(config.NEAREST_NEIGHBORS_PATH, index=False)
    
    # 6. Statistical Analysis
    analysis = analyze_embeddings(model.W_in, model.vocab)
    
    # 7. Visualizations
    print("Generating 6 Analytical Visualizations...")
    generate_visualizations(
        losses=model.epoch_losses,
        vocab=model.vocab,
        W=model.W_in,
        sim_matrix_df=sim_df,
        output_dir=config.CHARTS_DIR
    )
    
    # 8. Report
    print("Generating Markdown Report...")
    generate_report(
        analysis_data=analysis,
        top_neighbors_df=neighbors_df,
        losses=model.epoch_losses,
        output_path=config.REPORT_PATH
    )
    
    print("=== DAY 103 WORD2VEC ENGINE COMPLETE ===")

if __name__ == "__main__":
    main()
