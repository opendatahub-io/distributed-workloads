from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union, Tuple, Any
import numpy as np
import torch
from feast import FeatureView
from feast.online_response import OnlineResponse
from pyarrow import float16
from torch import dtype
from transformers import RagRetriever


class VectorStore(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int):
        pass


class FeastVectorStore(VectorStore):
    def __init__(self, store_repo_path: str, rag_view: FeatureView, features: List[str]):
        self._store = None  # Lazy load
        self._store_repo_path = store_repo_path
        self.rag_view = rag_view
        self.features = features

    @property
    def store(self):
        if self._store is None:
            from feast import FeatureStore
            self._store = FeatureStore(repo_path=self._store_repo_path)
            self._store.apply([self.rag_view])
        return self._store

    def query(
            self,
            query_vector: Optional[np.ndarray] = None,
            query_string: Optional[str] = None,
            top_k: int = 10,
    ) -> OnlineResponse:

        distance_metric = "COSINE" if query_vector is not None else None
        query_list = query_vector.tolist() if query_vector is not None else None

        return self.store.retrieve_online_documents_v2(
            features=self.features,
            query=query_list,
            query_string=query_string,
            top_k=top_k,
            distance_metric=distance_metric,
        )

# Dummy index - an index is required by the HF Transformers RagRetriever class
class FeastIndex:
    def __init__(self):
        pass


class FeastRAGRetriever(RagRetriever):
    VALID_SEARCH_TYPES = {"text", "vector", "hybrid"}

    def __init__(
            self,
            question_encoder_tokenizer,

            generator_tokenizer,
            feast_repo_path: str,
            search_type: str,
            feature_view: FeatureView,
            features: List[str],
            config,
            index,
            generator_model = None,
            question_encoder = None,
            format_document: Optional[Callable[[Dict], str]] = None,
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
        self.feast_repo_path = feast_repo_path
        self.search_type = search_type.lower()
        self.format_document = format_document or self._default_format_document
        self.retrieval_features = features
        self.feature_view = feature_view

        # Automatically determine the correct field names from the features list.
        self.text_field = self._get_field_name("text")
        self.embedding_field = self._get_field_name("embedding")
        self.id_field = self._get_field_name("id")


        # Initialize these lazily
        self._query_encoder = None
        self._feast_store = None
        self._vector_store = None

    @property
    def feast_store(self):
        # Initialize FeatureStore lazily
        if self._feast_store is None:
            from feast import FeatureStore
            self._feast_store = FeatureStore(repo_path=self.feast_repo_path)
        return self._feast_store

    @property
    def vector_store(self):
        # Initialize FeastVectorStore lazily
        if self._vector_store is None:
            self._vector_store = FeastVectorStore(
                store_repo_path=self.feast_repo_path,
                rag_view=self.feature_view,
                features=self.retrieval_features
            )
        return self._vector_store


    def retrieve(
            self,
            question_hidden_states: np.ndarray,
            n_docs: int,
            query: Optional[Union[str, List[str]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, List[str]]]]:
        """
        Overrides the base `retrieve` method to fetch documents from Feast.

        This method processes a batch of questions, queries the vector store,
        and formats the results into the 3-part tuple expected by the RAG model:
        (document_embeddings, document_ids, document_dictionaries).
        """
        batch_size = question_hidden_states.shape[0]

        # --- 1. Prepare Query Vectors ---
        # Convert the question hidden states into a list of 1D query vectors.
        pooled_query_vectors = []
        for i in range(batch_size):
            pooled = question_hidden_states[i]
            # Perform normalization to create a unit vector.
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
            pooled_query_vectors.append(pooled)

        # Determine the embedding dimension for padding, if needed
        emb_dim = pooled_query_vectors[0].shape[-1] if pooled_query_vectors else self.config.retrieval_vector_size

        # --- 2. Retrieve Documents for Each Query in the Batch ---
        batch_embeddings, batch_doc_ids, batch_metadata = [], [], []

        for i in range(batch_size):
            query_vector = pooled_query_vectors[i]
            query_text = query[i] if isinstance(query, list) and i < len(query) else query if isinstance(query, str) else None

            # Query Feast to get the raw document data
            response = self.vector_store.query(
                query_vector=query_vector if self.search_type != "text" else None,
                query_string=query_text if self.search_type != "vector" else None,
                top_k=n_docs
            )
            results_dict = response.to_dict()

            # --- 3. Process and Format the Retrieval Results ---
            # Dynamically get data using the configured feature names
            texts = results_dict.get(self.text_field, [])
            embeddings = results_dict.get(self.embedding_field, [])
            passage_ids = results_dict.get(self.id_field, [])
            num_retrieved = len(texts)

            # Initialize lists for the current query's results
            current_query_embeddings, current_query_ids, current_query_texts = [], [], []

            # Process retrieved documents up to n_docs
            emb_array = np.array
            if num_retrieved > 0:
                emb_array = np.array([np.asarray(emb, dtype=np.float32) for emb in embeddings])
                if emb_array.ndim == 1: # Reshape if it's a flat array
                    emb_array = emb_array.reshape(num_retrieved, -1)

            for k in range(n_docs):
                if k < num_retrieved:
                    # Append actual retrieved data
                    current_query_texts.append(texts[k])
                    current_query_embeddings.append(emb_array[k])
                    try:
                        current_query_ids.append(int(passage_ids[k]))
                    except (ValueError, TypeError):
                        current_query_ids.append(-1) # Use placeholder for invalid IDs
                else:
                    # Pad with empty/zero values if fewer than n_docs were found
                    current_query_texts.append("")
                    current_query_embeddings.append(np.zeros(emb_dim, dtype=np.float32))
                    current_query_ids.append(-1)

            # Collate results for the current query
            batch_embeddings.append(np.array(current_query_embeddings))
            batch_doc_ids.append(np.array(current_query_ids))
            batch_metadata.append({
                "text": current_query_texts,
                "id": current_query_ids,
                "title": [""] * n_docs  # RAG model expects a "title" key during Forward pass
            })

        contexts = batch_metadata[0]["text"] if batch_metadata else []
        context_str = "\n\n".join(filter(None, contexts))
        print("CONTEXl: ", context_str)
        # --- 4. Return the Collated Batch Results ---
        # Convert lists of arrays into single batch arrays.

        return (
            np.array(batch_embeddings, dtype=np.float32),
            np.array(batch_doc_ids, dtype=np.int64),
            batch_metadata
        )

    def is_question_answerable(
            self,
            question_text: str,
            expected_answer: str,
            question_encoder,
            tokenizer,
            top_k: int = 10,
            similarity_threshold: float = 0.6,
            max_answer_length: int = 50,
            min_answer_length: int = 1
    ) -> bool:
        
        # Basic answer validation
        expected_answer = expected_answer.strip().lower()
        answer_words = expected_answer.split()

        # Skip if answer is too short, too long, or empty
        if len(answer_words) < min_answer_length or len(answer_words) > max_answer_length:
            return False

        # Get question embeddings using the question encoder
        inputs = tokenizer(
            question_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(question_encoder.device)

        with torch.no_grad():
            # Get question hidden states
            question_outputs = question_encoder(**inputs)

            # Handle different model outputs
            if hasattr(question_outputs, 'pooler_output') and question_outputs.pooler_output is not None:
                question_hidden_states = question_outputs.pooler_output
            elif hasattr(question_outputs, 'last_hidden_state'):
                # Use mean pooling for models without pooler_output
                question_hidden_states = question_outputs.last_hidden_state.mean(dim=1)
            else:
                question_hidden_states = question_outputs[0].mean(dim=1)

            # Convert to numpy and ensure correct shape
            question_hidden_states = question_hidden_states.float().cpu().numpy()
            if question_hidden_states.ndim == 1:
                question_hidden_states = question_hidden_states.reshape(1, -1)

        # Retrieve docs from milvus
        doc_embeddings, doc_ids, doc_metadata = self.retrieve(
            question_hidden_states=question_hidden_states,
            n_docs=top_k,
            query=question_text
        )

        # Extract retrieved text from metadata
        retrieved_texts = []
        if doc_metadata and len(doc_metadata) > 0:
            metadata = doc_metadata[0]  # First (and only) query in batch
            if 'text' in metadata:
                retrieved_texts = [text.lower() for text in metadata['text'] if text.strip()]

        # Check if answer can be found in retrieved context
        if not retrieved_texts:
            return False

        # Combine all retrieved texts
        context_combined = ' '.join(retrieved_texts)

        # Multiple matching strategies
        exact_match = expected_answer in context_combined

        # Check for partial matches (significant words)
        answer_words_significant = [word for word in answer_words if len(word) > 2]
        partial_match = any(word in context_combined for word in answer_words_significant)

        # Word overlap similarity
        answer_word_set = set(answer_words)
        context_word_set = set(context_combined.split())
        word_overlap = len(answer_word_set.intersection(context_word_set)) / len(answer_word_set) if answer_word_set else 0

        # Decision criteria
        is_answerable = (
                exact_match or
                (partial_match and word_overlap > similarity_threshold) or
                word_overlap > 0.8
        )

        return is_answerable

    def generate_answer(
            self, query: str, top_k: int = 5, max_new_tokens: int = 100
    ) -> str:
        """A helper method to generate an answer for a single query string.

        Args:
            query: The query to answer
            top_k: Number of documents to retrieve
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated answer string
        """
        if not self.question_encoder or not self.generator_model:
            raise ValueError(
                "`question_encoder` and `generator_model` must be provided to use `generate_answer`."
            )
        # torch = get_torch()
        inputs = self.question_encoder_tokenizer(query, return_tensors="pt").to(
            self.question_encoder.device
        )
        question_embeddings = self.question_encoder(**inputs).pooler_output
        question_embeddings = (
            question_embeddings.detach().cpu().to(torch.float32).numpy()
        )
        _, _, doc_batch = self.retrieve(question_embeddings, n_docs=top_k, query=query)

        contexts = doc_batch[0]["text"] if doc_batch else []
        context_str = "\n\n".join(filter(None, contexts))

        prompt = f"""
        Answer the following question based only on the context provided.
        Context: {context_str}
        Question: {query}
        Answer:
        """

        generator_inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(
            self.generator_model.device
        )
        output_ids = self.generator_model.generate(
            **generator_inputs, max_new_tokens=max_new_tokens
        )

        return self.generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _get_field_name(self, keyword: str) -> str:
        """Finds the field name in the retrieval features that contains the specified keyword."""
        for feature in self.retrieval_features:
            if keyword in feature:
                return feature.split(':')[-1]
        raise ValueError(f"Could not find a feature containing the keyword '{keyword}' in the provided features list: {self.retrieval_features}")

    def _default_format_document(self, doc: Dict[str, Any]) -> str:
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