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
    def __init__(self, store: FeatureStore, rag_view: FeatureView):
        self.store = store
        self.rag_view = rag_view
        self.store.apply([rag_view])
        self.features = [f"{rag_view.name}:{field.name}" for field in rag_view.schema]

    def query(
        self,
        query_vector: np.ndarray,
        query_string: Optional[str] = "",
        top_k: int = 10,
    ):
        results = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=query_vector,
            query_string=query_string,
            top_k=top_k,
            distance_metric="COSINE",
        )
        return results

    def text_query(self, query: str, top_k: int = 10):
        results = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=None,
            query_string=query,
            top_k=top_k,
            distance_metric=None,
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
        if self.search_type == "text":
            return self._text_search(query, top_k)
        elif self.search_type == "vector":
            return self._vector_search(query, top_k)
        elif self.search_type == "hybrid":
            text_results = self._text_search(query, top_k)
            vector_results = self._vector_search(query, top_k)
            return self._merge_results(text_results, vector_results, top_k)
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")

    def _text_search(self, query: str, top_k: int):
        return self.vector_store.text_query(query, top_k)

    def _vector_search(self, query: str, top_k: int):
        query_vector_np = self.query_encoder.encode(query, convert_to_numpy=True)
        return self.vector_store.query(
            query_vector=query_vector_np, query_string="", top_k=top_k
        )

    def _merge_results(self, text_results, vector_results, top_k):
        # Combine, deduplicate by 'doc_id', and return up to top_k
        seen = set()
        combined = []

        def add_unique(results):
            for r in results:
                doc_id = r.get(self.id_field)
                if doc_id not in seen:
                    combined.append(r)
                    seen.add(doc_id)

        # Deduplicate and merge results from both text and vector queries
        add_unique(text_results)
        add_unique(vector_results)

        return combined[:top_k]

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
