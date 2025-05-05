import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
import re
from rnn import WordLevelReviewDataset, collate_fn

# Assurez-vous d'avoir les ressources nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Prétraitement du texte pour l'évaluation"""
    # Enlever la ponctuation et convertir en minuscules
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenization
    tokens = nltk.word_tokenize(text)
    return tokens

def calculate_rouge_score(generated_text, reference_text, n=1):
    """
    Calcule le score ROUGE-N entre le texte généré et la référence.
    
    Args:
        generated_text (str): Texte généré par le modèle
        reference_text (str): Texte de référence
        n (int): Taille des n-grammes à considérer
        
    Returns:
        dict: Dictionnaire contenant précision, rappel et F1 score
    """
    # Prétraitement
    gen_tokens = preprocess_text(generated_text)
    ref_tokens = preprocess_text(reference_text)
    
    # Calculer les n-grammes
    gen_ngrams = Counter(ngrams(gen_tokens, n))
    ref_ngrams = Counter(ngrams(ref_tokens, n))
    
    # Trouver les n-grammes communs
    common_ngrams = gen_ngrams & ref_ngrams
    
    # Calculer le nombre total de n-grammes dans chaque texte
    common_count = sum(common_ngrams.values())
    gen_count = sum(gen_ngrams.values()) or 1  # Éviter division par zéro
    ref_count = sum(ref_ngrams.values()) or 1  # Éviter division par zéro
    
    # Calculer précision, rappel et F1
    precision = common_count / gen_count if gen_count > 0 else 0
    recall = common_count / ref_count if ref_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_rouge_l(generated_text, reference_text):
    """
    Calcule le score ROUGE-L (Longest Common Subsequence) entre le texte généré et la référence.
    """
    # Prétraitement
    gen_tokens = preprocess_text(generated_text)
    ref_tokens = preprocess_text(reference_text)
    
    # Calculer la plus longue sous-séquence commune
    lcs_length = get_lcs_length(gen_tokens, ref_tokens)
    
    # Calculer précision, rappel et F1
    precision = lcs_length / len(gen_tokens) if gen_tokens else 0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def get_lcs_length(X, Y):
    """Calcule la longueur de la plus longue sous-séquence commune."""
    m, n = len(X), len(Y)
    L = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    return L[m][n]

def calculate_bleu_score(generated_text, reference_text, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calcule le score BLEU entre le texte généré et la référence.
    
    Args:
        generated_text (str): Texte généré par le modèle
        reference_text (str): Texte de référence
        weights (tuple): Poids pour les différents n-grammes (1-gram, 2-gram, etc.)
        
    Returns:
        float: Score BLEU
    """
    # Prétraitement
    gen_tokens = preprocess_text(generated_text)
    ref_tokens = preprocess_text(reference_text)
    
    # Fonction de lissage pour éviter les scores nuls
    smoothie = SmoothingFunction().method1
    
    # Calcul du score BLEU
    return sentence_bleu([ref_tokens], gen_tokens, weights=weights, smoothing_function=smoothie)

def load_word_vectors(path_to_vectors):
    """Charge des vecteurs de mots pré-entraînés."""
    try:
        return KeyedVectors.load_word2vec_format(path_to_vectors, binary=True)
    except Exception as e:
        print(f"Erreur lors du chargement des vecteurs: {e}")
        return None

def get_sentence_embedding(text, word_vectors):
    """
    Calcule l'embedding d'une phrase en moyennant les embeddings des mots.
    Si word_vectors est None, utilise une représentation one-hot simplifiée.
    """
    tokens = preprocess_text(text)
    
    if word_vectors is not None:
        # Utiliser des word vectors pré-entraînés
        embeddings = []
        for word in tokens:
            if word in word_vectors:
                embeddings.append(word_vectors[word])
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.zeros(word_vectors.vector_size)
    else:
        # Fallback: utiliser une représentation one-hot simplifiée
        vocabulary = list(set(tokens))
        embedding = np.zeros(len(vocabulary))
        for word in tokens:
            if word in vocabulary:
                embedding[vocabulary.index(word)] = 1
        return embedding / (np.sum(embedding) or 1)  # Normaliser

def calculate_semantic_similarity(generated_text, reference_text, word_vectors=None):
    """
    Calcule la similarité sémantique (cosinus) entre le texte généré et la référence.
    
    Args:
        generated_text (str): Texte généré par le modèle
        reference_text (str): Texte de référence
        word_vectors (KeyedVectors): Modèle de word embeddings pré-entraîné (optionnel)
        
    Returns:
        float: Score de similarité entre 0 et 1
    """
    # Obtenir les embeddings des phrases
    gen_embedding = get_sentence_embedding(generated_text, word_vectors)
    ref_embedding = get_sentence_embedding(reference_text, word_vectors)
    
    # Reshape pour le calcul de similarité cosinus
    gen_embedding = gen_embedding.reshape(1, -1)
    ref_embedding = ref_embedding.reshape(1, -1)
    
    # Calculer la similarité cosinus
    similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
    
    return max(0, min(similarity, 1))  # Borner entre 0 et 1

def evaluate_generated_text(generated_text, reference_text, word_vectors=None):
    """
    Évalue un texte généré en utilisant plusieurs métriques.
    
    Args:
        generated_text (str): Texte généré par le modèle
        reference_text (str): Texte de référence
        word_vectors (KeyedVectors): Modèle de word embeddings pré-entraîné (optionnel)
        
    Returns:
        dict: Dictionnaire contenant les scores des différentes métriques
    """
    results = {}
    
    # Calculer les scores ROUGE
    results["ROUGE-1"] = calculate_rouge_score(generated_text, reference_text, n=1)
    results["ROUGE-2"] = calculate_rouge_score(generated_text, reference_text, n=2)
    results["ROUGE-L"] = calculate_rouge_l(generated_text, reference_text)
    
    # Calculer le score BLEU
    results["BLEU"] = calculate_bleu_score(generated_text, reference_text)
    
    # Calculer la similarité sémantique
    results["Semantic_Similarity"] = calculate_semantic_similarity(generated_text, reference_text, word_vectors)
    
    return results

def evaluate_model_on_test_set(model, test_loader, vocab, references, word_vectors=None, device="cpu"):
    """
    Évalue le modèle sur un ensemble de test.
    
    Args:
        model: Le modèle à évaluer
        test_loader: DataLoader contenant les données de test
        vocab: Dictionnaire de vocabulaire
        references: Liste des textes de référence
        word_vectors: Modèle de word embeddings (optionnel)
        device: Appareil de calcul (CPU/GPU)
        
    Returns:
        dict: Moyennes des métriques d'évaluation
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i, (features, titles, _, _) in enumerate(test_loader):
            if i >= len(references):
                break
                
            features = features.to(device)
            titles = titles.to(device)
            
            # Générer un texte
            title_tensor = titles[0]  # Prendre le premier titre du batch
            features_tensor = features[0]  # Prendre les premières features du batch
            
            generated_text = model.generate(features_tensor, title_tensor, vocab)
            reference_text = references[i]
            
            # Évaluer le texte généré
            metrics = evaluate_generated_text(generated_text, reference_text, word_vectors)
            all_metrics.append(metrics)
            
            # Afficher les résultats pour cette instance
            print(f"\nÉvaluation de l'exemple {i+1}:")
            print(f"Texte généré: {generated_text}")
            print(f"Référence: {reference_text}")
            print(f"ROUGE-1 F1: {metrics['ROUGE-1']['f1']:.4f}")
            print(f"BLEU: {metrics['BLEU']:.4f}")
            print(f"Similarité sémantique: {metrics['Semantic_Similarity']:.4f}")
    
    # Calculer les moyennes des métriques
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], dict):
            avg_metrics[key] = {}
            for subkey in all_metrics[0][key].keys():
                avg_metrics[key][subkey] = sum(m[key][subkey] for m in all_metrics) / len(all_metrics)
        else:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics

# Exemple d'utilisation dans votre code
def evaluate_model(model, test_data, vocab, device="cpu"):
    """
    Fonction principale pour évaluer votre modèle RNN.
    
    Args:
        model: Votre modèle WordLevelReviewGenerator
        test_data: DataFrame contenant les données de test
        vocab: Dictionnaire de vocabulaire
        device: Appareil de calcul (CPU/GPU)
    """
    # Créer un dataset et dataloader de test
    test_dataset = WordLevelReviewDataset(test_data, vocab=vocab)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Extraire les références (textes réels) du dataset de test
    references = test_data['review'].tolist()
    
    # Option 1: Charger des word vectors pré-entraînés (recommandé pour la similarité sémantique)
    # word_vectors = load_word_vectors('path/to/vectors.bin')  # ex: GoogleNews-vectors-negative300.bin
    
    # Option 2: Utiliser None pour un fallback sur une représentation simplifiée
    word_vectors = None
    
    # Évaluer le modèle
    avg_metrics = evaluate_model_on_test_set(model, test_loader, vocab, references, word_vectors, device)
    
    # Afficher les résultats moyens
    print("\nRésultats moyens sur l'ensemble de test:")
    print(f"ROUGE-1 F1: {avg_metrics['ROUGE-1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {avg_metrics['ROUGE-2']['f1']:.4f}")
    print(f"ROUGE-L F1: {avg_metrics['ROUGE-L']['f1']:.4f}")
    print(f"BLEU: {avg_metrics['BLEU']:.4f}")
    print(f"Similarité sémantique: {avg_metrics['Semantic_Similarity']:.4f}")
    
    return avg_metrics

# Pour intégrer cette évaluation dans votre script principal, ajoutez ceci après l'entraînement:
"""
# Diviser les données pour l'évaluation
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Après avoir entraîné votre modèle:
print("Évaluation du modèle...")
eval_metrics = evaluate_model(model, test_df, train_dataset.word2idx, device)
"""
