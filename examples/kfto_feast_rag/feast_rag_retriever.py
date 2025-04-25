from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union, Any, Tuple

import numpy as np
from feast import FeatureStore, FeatureView
from sentence_transformers import SentenceTransformer
from transformers import RagRetriever


class VectorStore(ABC):
    @abstractmethod
    def query(
            self,
            query_vector: Optional[np.ndarray] = None,
            query_string: Optional[str] = None,
            top_k: int = 10,
        ) -> List[Dict[str, Any]]:        
        pass


class FeastVectorStore(VectorStore):
    def __init__(self, store: FeatureStore, rag_view: FeatureView, features: List[str]):
        self.store = store
        self.rag_view = rag_view
        self.store.apply([rag_view])
        self.features = features

    def query(
        self,
        query_vector: Optional[np.ndarray] = None,
        query_string: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
    
        distance_metric = "COSINE" if query_vector is not None else None
        query_list = query_vector.tolist() if query_vector is not None else None

        response = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=query_list,
            query_string=query_string,
            top_k=top_k,
            distance_metric=distance_metric,
        ).to_dict()
        
        results = []
        for feature_name in self.features:
            short_name = feature_name.split(":")[-1]
            feature_values = response[short_name]
            for i, value in enumerate(feature_values):
                if i >= len(results):
                    results.append({})
                results[i][short_name] = value
            
        return results


# Dummy index - an index is required by the HF Transformers RagRetriever class
class FeastIndex:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def get_top_docs(self, query_vectors: np.ndarray, n_docs: int = 5):
        raise NotImplementedError("get_top_docs is not yet implemented.")

    def get_doc_dicts(self, doc_ids: List[str]):
        raise NotImplementedError("get_doc_dicts is not yet implemented.")


class FeastRAGRetriever(RagRetriever):
    VALID_SEARCH_TYPES = {"text", "vector", "hybrid"}

    def __init__(
        self,
        question_encoder_tokenizer,
        question_encoder,
        generator_tokenizer,
        generator_model,
        feast_repo_path: str,
        vector_store: VectorStore,
        search_type: str,
        config: Dict[str, Any],
        index: FeastIndex,
        format_document: Optional[Callable[[Dict[str, Any]], str]] = None,
        id_field: str = "",
        query_encoder_model: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2",
        **kwargs,
    ):
        if search_type.lower() not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Unsupported search_type {search_type}. "
                f"Must be one of: {self.VALID_SEARCH_TYPES}"
            )
        super().__init__(
            config=config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
            init_retrieval=False,
            **kwargs,
        )
        self.question_encoder = question_encoder
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.feast = FeatureStore(repo_path=feast_repo_path)
        self.vector_store = vector_store
        self.search_type = search_type.lower()
        self.format_document = format_document or FeastRAGRetriever._default_format_document
        self.id_field = id_field

        if isinstance(query_encoder_model, str):
            self.query_encoder = SentenceTransformer(query_encoder_model)
        else:
            self.query_encoder = query_encoder_model

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int, query: Optional[str] = None) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        # Convert hidden states to query vector by pooling 
        query_vector = question_hidden_states.mean(dim=1).squeeze().detach().cpu().numpy()

        # Decode text query if needed (for hybrid or text search)
        if query is None and self.search_type in ("text", "hybrid"):
            query = self.question_encoder_tokenizer.decode(
                question_hidden_states.argmax(axis=-1),
                skip_special_tokens=True
            )
    
        if self.search_type == "text":
            results = self.vector_store.query(query_string=query, top_k=n_docs)
    
        elif self.search_type == "vector":
            results = self.vector_store.query(query_vector=query_vector, top_k=n_docs)
    
        elif self.search_type == "hybrid":
            results = self.vector_store.query(
                query_string=query,
                query_vector=query_vector,
                top_k=n_docs
            )
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")
    
        # Cosine similarity scoring
        doc_embeddings = np.array([doc["embedding"] for doc in results])
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    
        query_norm = np.maximum(query_norm, 1e-10)
        doc_norms = np.maximum(doc_norms, 1e-10)
    
        similarities = np.dot(doc_embeddings, query_vector) / (doc_norms * query_norm)
        doc_scores = similarities.reshape(1, -1)
        # passage_text is hardcoded at the moment
        doc_dicts = [{"text": doc["passage_text"]} for doc in results]
    
        return doc_scores, doc_dicts

    def generate_answer(
        self, query: str, top_k: int = 5, max_new_tokens: int = 100
    ) -> str:
        # Convert query to hidden states format expected by retrieve
        inputs = self.question_encoder_tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        )
        question_hidden_states = self.question_encoder(**inputs).last_hidden_state
        
        # Get documents using retrieve method
        doc_scores, doc_dicts = self.retrieve(question_hidden_states, n_docs=top_k)
        
        # Format context from retrieved documents
        contexts = [doc["text"] for doc in doc_dicts]
        context = "\n\n".join(contexts)
        
        prompt = (
            f"Use the following context to answer the question. Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        inputs = self.generator_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output_ids = self.generator_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.generator_tokenizer.pad_token_id,
        )
        return self.generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    @staticmethod
    def _default_format_document(doc: Dict[str, Any]) -> str:
        lines = []
        for key, value in doc.items():
            # Skip vectors by checking for long float lists
            if (
                isinstance(value, list)
                and len(value) > 10
                and all(isinstance(x, (float, int)) for x in value)
            ):
                continue
            lines.append(f"{key.replace('_', ' ').capitalize()}: {value}")
        return "\n".join(lines)
