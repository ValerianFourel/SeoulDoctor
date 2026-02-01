"""
========================================
RAG PIPELINE MODULE - HYBRID SEARCH
========================================

Handles all RAG (Retrieval-Augmented Generation) operations with hybrid search:
- ChromaDB vector database management
- BM25 keyword search
- Hybrid retrieval (BM25 + Vector with dynamic alpha)
- Query routing (FACTUAL vs MIXED)
- Document embedding and indexing
- Semantic search and ranking
- Context building for LLM generation
- Relevance scoring and result ranking
"""

import logging
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from rank_bm25 import BM25Okapi
import json

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Manages the RAG pipeline with hybrid search (BM25 + Vector) for semantic search 
    and ranking of medical facilities.
    
    Features:
    - Vector database initialization and management
    - BM25 keyword indexing
    - Hybrid search with dynamic alpha routing
    - Query classification (FACTUAL vs MIXED)
    - Document embedding and indexing
    - Semantic similarity search
    - Adaptive relevance scoring (distance + semantic similarity)
    - Context generation for LLM responses
    """
    
    def __init__(
        self, 
        chroma_path: str,
        openai_api_key: str,
        groq_client,  # For query routing
        collection_name: str = "seoul_med_v3",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the RAG pipeline with hybrid search capabilities.
        
        Args:
            chroma_path: Path to ChromaDB persistent storage
            openai_api_key: OpenAI API key for embeddings
            groq_client: Groq client for query routing LLM
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.vector_db = None
        self.groq_client = groq_client
        
        # BM25 components
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model
        )
        
        logger.info(f"✅ RAG Pipeline initialized at {chroma_path}")
    
    def initialize_collection(self, df_filtered: pd.DataFrame, force_recreate: bool = False) -> None:
        """
        Initialize or load the vector database collection and BM25 index.
        
        Args:
            df_filtered: Filtered DataFrame with facility data
            force_recreate: If True, delete and recreate the collection
        """
        try:
            if force_recreate:
                # Delete existing collection if it exists
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"🗑️ Deleted existing collection '{self.collection_name}'")
                except:
                    pass
            
            # Try to get existing collection
            try:
                self.vector_db = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"✅ Loaded existing collection '{self.collection_name}'")
                
                # Verify collection has data
                count = len(self.vector_db.get()['ids'])
                logger.info(f"   Collection contains {count} documents")
                
                if count == 0:
                    logger.warning("⚠️ Collection is empty, will rebuild")
                    raise Exception("Empty collection")
                
            except Exception as e:
                logger.info(f"⚠️ Creating new collection: {e}")
                self._create_and_index_collection(df_filtered)
            
            # Build BM25 index
            self._build_bm25_index(df_filtered)
                
        except Exception as e:
            logger.error(f"❌ Error initializing collection: {e}", exc_info=True)
            raise
    
    def _create_and_index_collection(self, df_filtered: pd.DataFrame) -> None:
        """
        Create a new collection and index all documents.
        
        Args:
            df_filtered: Filtered DataFrame with facility data
        """
        logger.info("🔨 Creating new embeddings from filtered dataset...")
        
        # Create new collection
        self.vector_db = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        # Prepare documents for indexing
        ids, docs, metas = self._prepare_documents_for_indexing(df_filtered)
        
        # Batch insert documents
        batch_size = 100
        total_batches = (len(ids) + batch_size - 1) // batch_size
        
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            batch_num = (i // batch_size) + 1
            
            self.vector_db.add(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metas[i:end]
            )
            
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"   Indexed batch {batch_num}/{total_batches} ({end}/{len(ids)} documents)")
        
        logger.info(f"✅ Embeddings indexed: {len(ids)} documents")
    
    def _build_bm25_index(self, df_filtered: pd.DataFrame) -> None:
        """
        Build BM25 keyword index from documents.
        
        Args:
            df_filtered: Filtered DataFrame with facility data
        """
        logger.info("🔨 Building BM25 keyword index...")
        
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        
        for _, row in df_filtered.iterrows():
            # Extract text for BM25 (same as vector embedding text)
            summary_en = self._extract_array_field(row, 'Summaries', default="")
            summary_kr = self._extract_array_field(row, 'Summaries_Korean', default="")
            
            highlights = ""
            if 'Key_Highlights' in row.index and isinstance(row['Key_Highlights'], (list, np.ndarray)):
                if len(row['Key_Highlights']) > 0:
                    highlight_topics = [
                        h.get('topic', '') 
                        for h in row['Key_Highlights'] 
                        if isinstance(h, dict)
                    ]
                    highlights = ", ".join(highlight_topics)
            
            # Build text blob
            text_blob = f"{row['name']} {row['category']} {summary_en} {summary_kr} {highlights}"
            
            # Add to corpus (tokenized for BM25)
            tokenized = text_blob.lower().split()
            self.bm25_corpus.append(tokenized)
            self.bm25_doc_ids.append(str(row['place_id']))
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"✅ BM25 index built: {len(self.bm25_corpus)} documents")
    
    def _prepare_documents_for_indexing(
        self, 
        df_filtered: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Prepare documents from DataFrame for indexing.
        
        Args:
            df_filtered: Filtered DataFrame with facility data
            
        Returns:
            Tuple of (ids, documents, metadatas)
        """
        ids = []
        docs = []
        metas = []
        
        for _, row in df_filtered.iterrows():
            # Extract summaries
            summary_en = self._extract_array_field(row, 'Summaries', default="")
            summary_kr = self._extract_array_field(row, 'Summaries_Korean', default="")
            
            # Extract highlights
            highlights = ""
            if 'Key_Highlights' in row.index and isinstance(row['Key_Highlights'], (list, np.ndarray)):
                if len(row['Key_Highlights']) > 0:
                    highlight_topics = [
                        h.get('topic', '') 
                        for h in row['Key_Highlights'] 
                        if isinstance(h, dict)
                    ]
                    highlights = ", ".join(highlight_topics)
            
            # Build rich text blob for embedding
            text_blob = f"{row['name']} ({row['category']}). {summary_en} {summary_kr} {highlights}"
            
            # Store document
            ids.append(str(row['place_id']))
            docs.append(text_blob)
            
            # Store metadata
            metas.append({
                "category": str(row['category']),
                "district": str(row.get('file_district', '')),
                "has_english": bool(row.get('has_english', False))
            })
        
        return ids, docs, metas
    
    @staticmethod
    def _extract_array_field(row: pd.Series, field_name: str, default: str = "") -> str:
        """
        Safely extract first element from array field.
        
        Args:
            row: DataFrame row
            field_name: Name of the field to extract
            default: Default value if extraction fails
            
        Returns:
            Extracted string value or default
        """
        if field_name in row.index:
            field_value = row[field_name]
            if isinstance(field_value, (list, np.ndarray)) and len(field_value) > 0:
                return str(field_value[0])
        return default
    
    def route_query(self, query_text: str) -> Dict[str, Any]:
        """
        Route query to determine search strategy (FACTUAL vs MIXED).
        
        Args:
            query_text: User's search query
            
        Returns:
            Dictionary with 'intent', 'suggested_alpha', 'reasoning'
        """
        from prompt import QUERY_ROUTER_PROMPT,SPECIALTY_MAPPING
        
        if not self.groq_client:
            logger.warning("⚠️ No Groq client, defaulting to MIXED search")
            return {
                "intent": "MIXED",
                "suggested_alpha": 0.7,
                "reasoning": "No router available, using default mixed search"
            }
        
        try:
            router_prompt = QUERY_ROUTER_PROMPT.format(SPECIALTY_MAPPING=SPECIALTY_MAPPING, query=query_text)

            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": router_prompt}],
                temperature=0.0,
                max_completion_tokens=256,
                response_format={"type": "json_object"}
            )
            
            route_decision = json.loads(completion.choices[0].message.content)
            
            logger.debug(
                f"🎯 Query router: {route_decision['intent']} "
                f"(α={route_decision['suggested_alpha']:.2f}) - {route_decision['reasoning']}"
            )
            
            return route_decision
            
        except Exception as e:
            logger.error(f"❌ Query routing error: {e}", exc_info=True)
            return {
                "intent": "MIXED",
                "suggested_alpha": 0.7,
                "reasoning": "Router error, using default"
            }
    
    def calculate_alpha(
        self, 
        route_decision: Dict[str, Any], 
        manual_mode: Optional[str] = None
    ) -> float:
        """
        Calculate alpha for hybrid search with clipping logic.
        
        Alpha controls the balance between keyword (BM25) and semantic (vector) search:
        - 0.0 = Pure keyword (BM25 only)
        - 1.0 = Pure semantic (vector only)
        
        Clipping Logic:
        - Minimum alpha = 0.3 (prevents keyword tunnel vision)
        - Maximum alpha = 1.0 (allows full semantic search)
        
        Args:
            route_decision: Decision from route_query with 'suggested_alpha'
            manual_mode: Optional manual override ('FACTUAL_ONLY' or 'MIXED')
            
        Returns:
            Clipped alpha value between 0.3 and 1.0
        """
        suggested_alpha = route_decision.get('suggested_alpha', 0.7)
        
        if manual_mode == "FACTUAL_ONLY":
            # Force into factual range (0.3-0.5)
            alpha = max(0.3, min(suggested_alpha, 0.5))
            logger.debug(f"🔧 Manual FACTUAL mode: α={alpha:.2f}")
            return alpha
        
        elif manual_mode == "MIXED":
            # Force into mixed range (0.6-0.8)
            alpha = max(0.6, min(suggested_alpha, 0.8))
            logger.debug(f"🔧 Manual MIXED mode: α={alpha:.2f}")
            return alpha
        
        # Auto mode: apply clipping (0.3-1.0)
        alpha = max(0.3, min(suggested_alpha, 1.0))
        
        logger.debug(
            f"🎯 Auto mode: α={alpha:.2f} "
            f"(suggested: {suggested_alpha:.2f}, clipped to 0.3-1.0 range)"
        )
        
        return alpha
    
    def bm25_search(self, query_text: str, n_results: int = 200) -> Dict[str, List]:
        """
        Perform BM25 keyword search.
        
        Args:
            query_text: Search query
            n_results: Maximum number of results
            
        Returns:
            Dictionary with 'ids' and 'scores'
        """
        if not self.bm25:
            logger.warning("⚠️ BM25 index not initialized")
            return {"ids": [], "scores": []}
        
        # Tokenize query
        tokenized_query = query_text.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top N results
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        result_ids = [self.bm25_doc_ids[i] for i in top_indices]
        result_scores = [scores[i] for i in top_indices]
        
        # Filter out zero scores
        filtered_results = [
            (doc_id, score) 
            for doc_id, score in zip(result_ids, result_scores) 
            if score > 0
        ]
        
        if filtered_results:
            result_ids, result_scores = zip(*filtered_results)
        else:
            result_ids, result_scores = [], []
        
        logger.debug(f"🔍 BM25 search: {len(result_ids)} results")
        
        return {
            "ids": list(result_ids),
            "scores": list(result_scores)
        }
    
    def vector_search(self, query_text: str, n_results: int = 200) -> Optional[Dict[str, Any]]:
        """
        Perform pure vector similarity search.
        
        Args:
            query_text: Search query
            n_results: Maximum number of results to return
            
        Returns:
            Dictionary with 'ids', 'distances', 'metadatas' or None if search fails
        """
        if not self.vector_db:
            logger.error("❌ Vector database not initialized")
            return None
        
        if not query_text or len(query_text.strip()) < 2:
            logger.warning("⚠️ Query too short for semantic search")
            return None
        
        try:
            # Limit n_results to available documents
            available_docs = len(self.vector_db.get()['ids'])
            n_results = min(n_results, available_docs)
            
            # Perform query
            results = self.vector_db.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            if results and 'ids' in results and len(results['ids']) > 0:
                logger.debug(f"🔍 Vector search: {len(results['ids'][0])} results")
                return results
            else:
                logger.warning("⚠️ No results from vector search")
                return None
                
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}", exc_info=True)
            return None
    
    def hybrid_search(
        self, 
        query_text: str, 
        alpha: float = 0.7, 
        n_results: int = 200
    ) -> Optional[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query_text: Search query
            alpha: Balance between keyword and semantic (0.0=keyword, 1.0=semantic)
            n_results: Maximum number of results
            
        Returns:
            Dictionary with combined results
        """
        logger.debug(f"🔀 Hybrid search: α={alpha:.2f} (keyword={(1-alpha):.0%}, semantic={alpha:.0%})")
        
        # Get BM25 results
        bm25_results = self.bm25_search(query_text, n_results)
        
        # Get vector results
        vector_results = self.vector_search(query_text, n_results)
        
        if not bm25_results['ids'] and not vector_results:
            logger.warning("⚠️ No results from either search method")
            return None
        
        # Combine scores
        combined_scores = {}
        
        # Add BM25 scores (weighted by 1-alpha)
        if bm25_results['ids']:
            max_bm25 = max(bm25_results['scores']) if bm25_results['scores'] else 1.0
            for doc_id, score in zip(bm25_results['ids'], bm25_results['scores']):
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                combined_scores[doc_id] = (1 - alpha) * normalized_score
        
        # Add vector scores (weighted by alpha)
        if vector_results and 'ids' in vector_results:
            # ChromaDB returns distances (lower is better), convert to similarity
            max_distance = max(vector_results['distances'][0]) if vector_results['distances'][0] else 1.0
            
            for doc_id, distance in zip(vector_results['ids'][0], vector_results['distances'][0]):
                # Convert distance to similarity (1 - normalized_distance)
                similarity = 1 - (distance / max_distance) if max_distance > 0 else 0
                
                if doc_id in combined_scores:
                    combined_scores[doc_id] += alpha * similarity
                else:
                    combined_scores[doc_id] = alpha * similarity
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_results]
        
        result_ids = [doc_id for doc_id, _ in sorted_results]
        result_scores = [score for _, score in sorted_results]
        
        logger.debug(
            f"✓ Hybrid search: {len(result_ids)} results "
            f"(BM25: {len(bm25_results['ids'])}, Vector: {len(vector_results['ids'][0]) if vector_results else 0})"
        )
        
        return {
            "ids": [result_ids],  # Wrap in list to match ChromaDB format
            "scores": [result_scores],
            "method": "hybrid",
            "alpha": alpha
        }
    
    def semantic_search(
        self, 
        query_text: str, 
        n_results: int = 200,
        use_hybrid: bool = True,
        manual_mode: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Perform semantic search with optional hybrid mode and query routing.
        
        Args:
            query_text: Search query
            n_results: Maximum number of results to return
            use_hybrid: If True, use hybrid search with query routing
            manual_mode: Optional manual override ('FACTUAL_ONLY' or 'MIXED')
            
        Returns:
            Dictionary with 'ids', 'distances'/'scores', 'metadatas' or None if search fails
        """
        if not query_text or len(query_text.strip()) < 2:
            logger.warning("⚠️ Query too short for search")
            return None
        
        if not use_hybrid:
            # Fall back to pure vector search
            return self.vector_search(query_text, n_results)
        
        # Route query to determine search strategy
        route_decision = self.route_query(query_text)
        
        # Calculate alpha with clipping
        alpha = self.calculate_alpha(route_decision, manual_mode)
        
        # Perform hybrid search
        return self.hybrid_search(query_text, alpha, n_results)
    
    def apply_semantic_ranking(
        self,
        df: pd.DataFrame,
        query_text: str,
        n_results: int = 200,
        use_hybrid: bool = True,
        manual_mode: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply semantic ranking to a DataFrame using RAG (with optional hybrid search).
        
        Args:
            df: DataFrame to rank
            query_text: Search query for semantic matching
            n_results: Number of RAG results to retrieve
            use_hybrid: If True, use hybrid search with query routing
            manual_mode: Optional manual override for search mode
            
        Returns:
            DataFrame with 'relevance_rank' column added
        """
        if len(query_text.strip()) < 2 or len(df) == 0:
            df['relevance_rank'] = 9999
            return df
        
        # Perform semantic search (hybrid or pure vector)
        rag_results = self.semantic_search(
            query_text, 
            n_results, 
            use_hybrid=use_hybrid,
            manual_mode=manual_mode
        )
        
        if not rag_results or 'ids' not in rag_results or len(rag_results['ids']) == 0:
            logger.debug("⚠️ No RAG results, using default ranking")
            df['relevance_rank'] = 9999
            return df
        
        # Create ranking dictionary (lower rank = more relevant)
        rag_ranking = {
            place_id: idx 
            for idx, place_id in enumerate(rag_results['ids'][0])
        }
        
        # Apply ranking to DataFrame
        df['relevance_rank'] = df['place_id'].apply(
            lambda pid: rag_ranking.get(pid, 9999)
        )
        
        # Log ranking statistics
        ranked_count = (df['relevance_rank'] < 9999).sum()
        
        search_method = rag_results.get('method', 'vector')
        alpha_info = f" (α={rag_results.get('alpha', 'N/A'):.2f})" if search_method == 'hybrid' else ""
        
        logger.debug(f"✓ RAG ranked: {ranked_count}/{len(df)} facilities [{search_method}{alpha_info}]")
        
        return df
    
    def apply_combined_ranking(
        self,
        df: pd.DataFrame,
        query_text: str,
        max_distance: float,
        search_mode: str = 'distance',
        n_results: int = 200,
        use_hybrid: bool = True,
        manual_mode: Optional[str] = None,
        is_general_search: bool = False,  # ⭐ NEW: Flag for general/random searches
        specialty_confidence: float = 1.0  # ⭐ NEW: Specialty confidence score
    ) -> pd.DataFrame:
        """
        Apply combined ranking using both semantic similarity and distance.
        Uses adaptive weighting based on search radius and specialty confidence.
        
        Args:
            df: DataFrame to rank
            query_text: Search query for semantic matching
            max_distance: Maximum search distance in km
            search_mode: 'distance' or 'zone'
            n_results: Number of RAG results to retrieve
            use_hybrid: If True, use hybrid search with query routing
            manual_mode: Optional manual override for search mode
            is_general_search: If True, skip semantic ranking (distance-only)
            specialty_confidence: Confidence in specialty (0.0-1.0), affects weighting
            
        Returns:
            DataFrame sorted by combined score (or distance for general searches)
        """
        
        # ===== GENERAL SEARCH: Distance-only ranking =====
        if is_general_search or specialty_confidence < 0.3:
            logger.debug("🎲 General search mode → distance-only ranking (no semantic filtering)")
            
            if 'distance_km' in df.columns and df['distance_km'].max() > 0:
                # Sort by distance (closest first)
                df = df.sort_values('distance_km', ascending=True)
                
                # Add placeholder scores for consistency
                df['relevance_rank'] = 9999
                df['relevance_score'] = 0.0
                df['distance_score'] = 1.0
                df['combined_score'] = df['distance_score']
                
                logger.debug(f"✓ General search: {len(df)} facilities sorted by distance")
            else:
                # No distance data, keep original order
                logger.warning("⚠️ No distance data available for general search")
            
            return df
        
        # ===== SPECIALTY SEARCH: Semantic + Distance ranking =====
        
        # Apply semantic ranking (with hybrid search)
        df = self.apply_semantic_ranking(
            df, 
            query_text, 
            n_results,
            use_hybrid=use_hybrid,
            manual_mode=manual_mode
        )
        
        # If zone-based search, just sort by relevance
        if search_mode == 'zone' or 'distance_km' not in df.columns:
            logger.debug("🏘️ Zone search → relevance-only ranking")
            return df.sort_values('relevance_rank')
        
        # If no distance data, sort by relevance only
        if df['distance_km'].max() == 0:
            logger.debug("⚠️ No distance data → relevance-only ranking")
            return df.sort_values('relevance_rank')
        
        # ===== COMBINED SCORING: Distance + Semantic Relevance =====
        
        # Calculate normalized scores
        max_rank = df['relevance_rank'].max()
        max_dist = df['distance_km'].max()
        
        # Relevance score (higher is better)
        if max_rank > 0:
            df['relevance_score'] = 1 - (df['relevance_rank'] / max_rank)
        else:
            df['relevance_score'] = 1.0
        
        # Distance score (higher is better = closer)
        if max_dist > 0:
            df['distance_score'] = 1 - (df['distance_km'] / max_dist)
        else:
            df['distance_score'] = 1.0
        
        # ⭐ Adaptive weighting based on BOTH radius AND specialty confidence
        relevance_weight = self._calculate_relevance_weight(
            max_distance, 
            specialty_confidence=specialty_confidence
        )
        distance_weight = 1 - relevance_weight
        
        # Combined score
        df['combined_score'] = (
            relevance_weight * df['relevance_score'] + 
            distance_weight * df['distance_score']
        )
        
        # Sort by combined score (higher is better)
        df = df.sort_values('combined_score', ascending=False)
        
        logger.debug(
            f"✓ Combined ranking: {relevance_weight:.0%} relevance + "
            f"{distance_weight:.0%} distance "
            f"(radius={max_distance}km, spec_conf={specialty_confidence:.2f})"
        )
        
        return df

    @staticmethod
    def _calculate_relevance_weight(
        max_distance: float, 
        specialty_confidence: float = 1.0
    ) -> float:
        """
        Calculate adaptive relevance weight - KEYWORD-FOCUSED VERSION.
        
        Heavily favors keyword/semantic matching over pure distance.
        
        Args:
            max_distance: Maximum search distance in km
            specialty_confidence: Confidence in specialty match (0.0-1.0)
            
        Returns:
            Relevance weight (0.0 to 1.0)
        """
        
        # ===== STEP 1: Calculate base relevance weight (INCREASED FROM BEFORE) =====
        if max_distance <= 2:
            base_relevance_weight = 0.70  # Was 0.50, now 0.70 (keywords > distance even nearby)
        elif max_distance <= 5:
            # Linear interpolation between 2-5km
            base_relevance_weight = 0.70 + (max_distance - 2) * (0.85 - 0.70) / (5 - 2)
        elif max_distance <= 10:
            # Linear interpolation between 5-10km
            base_relevance_weight = 0.85 + (max_distance - 5) * (0.95 - 0.85) / (10 - 5)
        else:
            # Cap at 98% for very large radii (keywords almost completely dominate)
            base_relevance_weight = min(0.98, 0.95 + (max_distance - 10) * 0.005)
        
        # ===== STEP 2: Adjust based on specialty confidence (LESS aggressive reduction) =====
        if specialty_confidence < 0.4:
            # Very low confidence -> still favor keywords (reduced penalty: was 0.60, now 0.75)
            confidence_multiplier = 0.75
            logger.debug(f"   Low confidence adjustment: {confidence_multiplier:.0%} multiplier")
        elif specialty_confidence < 0.7:
            # Medium confidence -> keywords still strong (reduced penalty: was 0.80, now 0.90)
            confidence_multiplier = 0.90
            logger.debug(f"   Medium confidence adjustment: {confidence_multiplier:.0%} multiplier")
        else:
            # High confidence -> trust semantic similarity fully
            confidence_multiplier = 1.0
        
        # Apply confidence adjustment
        adjusted_relevance_weight = base_relevance_weight * confidence_multiplier
        
        # NEW: Minimum keyword consideration (never go below 50% keyword weight)
        final_relevance_weight = max(0.50, min(0.98, adjusted_relevance_weight))
        
        logger.debug(
            f"   Relevance weight: {final_relevance_weight:.2f} (KEYWORD-FOCUSED) "
            f"(base: {base_relevance_weight:.2f}, conf_mult: {confidence_multiplier:.2f})"
        )
        
        return final_relevance_weight
        
    def build_context_for_llm(
        self, 
        df_subset: pd.DataFrame, 
        n_results: int = 10,
        language: str = "English"
    ) -> str:
        """
        Build rich context from filtered facilities for LLM generation.
        
        Args:
            df_subset: DataFrame subset with top results
            n_results: Number of facilities to include in context
            language: Language preference for summaries
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        for idx, row in df_subset.head(n_results).iterrows():
            facility_info = []
            
            # Basic info
            facility_info.append(f"Name: {row['name']}")
            facility_info.append(f"Category: {row['category']}")
            
            # Location info
            if 'file_district' in row.index and pd.notna(row['file_district']):
                facility_info.append(f"District: {row['file_district']}")
            if 'file_dong' in row.index and pd.notna(row['file_dong']):
                facility_info.append(f"Neighborhood: {row['file_dong']}")
            
            # Distance info
            if 'distance_km' in row.index and pd.notna(row['distance_km']) and row['distance_km'] > 0:
                facility_info.append(f"Distance: {row['distance_km']:.1f}km away")
            
            # Summaries (prefer language-specific)
            if language == "Korean" and 'Summaries_Korean' in row.index:
                summaries_kr = row['Summaries_Korean']
                if isinstance(summaries_kr, (list, np.ndarray)) and len(summaries_kr) > 0:
                    facility_info.append(f"Summary: {summaries_kr[0]}")
            
            if language == "English" and 'Summaries' in row.index:
                summaries_en = row['Summaries']
                if isinstance(summaries_en, (list, np.ndarray)) and len(summaries_en) > 0:
                    facility_info.append(f"Summary: {summaries_en[0]}")
            
            # Highlights
            if 'Key_Highlights' in row.index:
                highlights_data = row['Key_Highlights']
                if isinstance(highlights_data, (list, np.ndarray)) and len(highlights_data) > 0:
                    topics = []
                    for h in highlights_data[:5]:
                        if isinstance(h, dict):
                            topic_key = 'topic' if language == "English" else 'topic'
                            if topic_key in h:
                                topics.append(h[topic_key])
                    if topics:
                        facility_info.append(f"Highlights: {', '.join(topics)}")
            
            # English support
            if 'has_english' in row.index and row['has_english']:
                facility_info.append("English Speaking: Yes")
            
            context_parts.append("\n".join(facility_info))
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "initialized": self.vector_db is not None,
            "collection_name": self.collection_name,
            "document_count": 0,
            "bm25_indexed": self.bm25 is not None,
            "bm25_document_count": len(self.bm25_corpus) if self.bm25 else 0,
            "hybrid_search_enabled": self.bm25 is not None and self.vector_db is not None,
            "chroma_path": self.chroma_path
        }
        
        if self.vector_db:
            try:
                stats["document_count"] = len(self.vector_db.get()['ids'])
            except:
                pass
        
        return stats