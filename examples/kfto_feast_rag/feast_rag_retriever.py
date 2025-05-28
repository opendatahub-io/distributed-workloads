from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from feast import FeatureStore, FeatureView
from sentence_transformers import SentenceTransformer
from transformers import RagRetriever


class VectorStore(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int):
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
    ):
        distance_metric = "COSINE" if query_vector is not None else None

        results = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=query_vector,
            query_string=query_string,
            top_k=top_k,
            distance_metric=distance_metric,
        )
        return results


# Dummy index - an index is required by the HF Transformers RagRetriever class
class FeastIndex:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_top_docs(self, query_vectors, n_docs=5):
        raise NotImplementedError("get_top_docs is not yet implemented.")

    def get_doc_dicts(self, doc_ids):
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
        config,
        index,
        format_document: Optional[Callable[[Dict], str]] = None,
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
        self.format_document = format_document or self._default_format_document
        self.id_field = id_field

        if isinstance(query_encoder_model, str):
            self.query_encoder = SentenceTransformer(query_encoder_model)
        else:
            self.query_encoder = query_encoder_model

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vector_np = self.query_encoder.encode(query, convert_to_numpy=True)

        if self.search_type == "text":
            return self.vector_store.query(query_string=query, top_k=top_k)        
        elif self.search_type == "vector":
            return self.vector_store.query(query_vector=query_vector_np, query_string="", top_k=top_k)
        elif self.search_type == "hybrid":
            return self.vector_store.query(
                query_string=query,
                query_vector=query_vector_np,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")

    def generate_answer(
        self, query: str, top_k: int = 5, max_new_tokens: int = 100
    ) -> str:
        # Retrieve top-k relevant documents
        documents = self.retrieve(query, top_k=top_k)
        document_dict = documents.to_dict()

        num_results = len(document_dict["passage_text"])
        contexts = []
        for i in range(num_results):
            passage_text = document_dict["passage_text"][i]
            contexts.append(passage_text)

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

    def _default_format_document(self, doc: dict) -> str:
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
