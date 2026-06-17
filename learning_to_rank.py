import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import re


class LearningToRank:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.feature_names = []
        self.is_trained = False
    
    def extract_features(self, query, document):
        """Extract features from query-document pair."""
        query_tokens = set(re.findall(r'\b[a-z0-9]+\b', query.lower()))
        doc_tokens = set(re.findall(r'\b[a-z0-9]+\b', document.lower()))
        
        # Common features
        query_length = len(query_tokens)
        doc_length = len(doc_tokens)
        
        # Term overlap features
        common_terms = query_tokens.intersection(doc_tokens)
        overlap_count = len(common_terms)
        overlap_ratio = overlap_count / max(query_length, 1)
        
        # Feature 1: Query-document term overlap ratio
        # Feature 2: Document length (log scaled)
        # Feature 3: Query length
        # Feature 4: Number of common terms
        # Feature 5: Common term ratio normalized by query length
        # Feature 6: Common term ratio normalized by document length
        
        features = [
            overlap_ratio,
            np.log(doc_length + 1),
            min(query_length / 10, 1.0),
            overlap_count / max(query_length, 1),
            overlap_count / max(doc_length, 1),
            overlap_count / max(overlap_count + 1, 1)
        ]
        
        return features
    
    def train(self, query_doc_pairs, relevance_scores):
        """Train the LTR model with query-document pairs and relevance scores."""
        feature_vectors = []
        
        for query, doc in query_doc_pairs:
            features = self.extract_features(query, doc)
            feature_vectors.append(features)
        
        X = np.array(feature_vectors)
        y = np.array(relevance_scores)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_score(self, query, document):
        """Predict relevance score for a query-document pair."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = np.array(self.extract_features(query, document)).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]
    
    def rank_documents(self, query, documents):
        """Rank documents by predicted relevance score."""
        results = []
        for doc_id, doc_text in documents:
            score = self.predict_score(query, doc_text)
            results.append((doc_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Example usage
ltr = LearningToRank()

# Training data: (query, document) pairs with relevance scores
training_pairs = [
    ("quick fox", "The quick brown fox jumps"),
    ("quick fox", "A slow fox walks"),
    ("quick fox", "The dog runs quickly"),
    ("lazy dog", "The lazy dog sleeps"),
    ("lazy dog", "An active dog plays"),
    ("lazy dog", "The fox is lazy"),
    ("brown fox", "The brown fox is quick"),
    ("brown fox", "The red fox is brown")
]

relevance_scores = [5, 3, 2, 5, 2, 3, 5, 3]  # Ground truth relevance

ltr.train(training_pairs, relevance_scores)

# Test ranking
test_query = "quick fox"
test_documents = [
    (1, "The quick brown fox runs fast"),
    (2, "A lazy dog sleeps all day"),
    (3, "The fox is very quick")
]

ranked_results = ltr.rank_documents(test_query, test_documents)
print(f"Ranked results for '{test_query}':", ranked_results)