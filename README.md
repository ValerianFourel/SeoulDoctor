---
title: Seoul Doctor Matchmaker
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Seoul Doctor Matchmaker 🏥

**AI-Powered Medical Facility Search for Seoul, South Korea**

Public, launch-ready documentation is available at [`/docs`](http://localhost:3000/docs) when the frontend is running.

A sophisticated conversational AI system that helps users find the right medical facilities in Seoul based on their specific needs, location, and preferences. The system uses hybrid search (BM25 + Vector embeddings), natural language processing, and intelligent routing to provide highly personalized medical facility recommendations.

Deploy the complete frontend and API as one Docker Space by following [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Core Components](#core-components)
- [Search Strategy](#search-strategy)
- [Keyword System](#keyword-system)
- [RAG Pipeline](#rag-pipeline)
- [Data Pipeline](#data-pipeline)
- [Conversation Flow](#conversation-flow)
- [Development](#development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 🎯 Overview

Seoul Doctor Matchmaker is an intelligent medical facility search system designed to overcome the language barrier and information asymmetry that foreign residents face when seeking healthcare in Seoul. The system processes natural language queries in both English and Korean, understands user intent, extracts search criteria, and returns highly relevant medical facilities with detailed information.

### Problem Statement

Foreign residents in Seoul face several challenges:
- **Language Barrier**: Most medical facilities have limited English-language information
- **Information Overload**: Thousands of medical facilities in Seoul with varying specialties
- **Preference Matching**: Hard to find facilities matching specific criteria (parking, insurance, English-speaking staff)
- **Location Complexity**: Seoul's 25 districts and 424 neighborhoods create geographical challenges

### Solution

An AI-powered conversational interface that:
- Understands natural language queries in English and Korean
- Extracts and validates search criteria (specialty, location, keywords)
- Uses hybrid search (keyword + semantic) for maximum relevance
- Provides rich context including reviews, amenities, and navigation
- Supports unconventional preferences (e.g., "unfriendly but efficient doctor")

---

## ✨ Key Features

### 🤖 Conversational AI
- **Natural Language Understanding**: Processes queries like "find me a friendly dentist with parking in Gangnam"
- **Intent Detection**: 9 conversation intents (PROVIDE_INFO, CHANGE_CRITERIA, EMERGENCY, etc.)
- **Multi-turn Dialogue**: Remembers context across conversation turns
- **State Management**: Tracks specialty, location, keywords, and search history
- **Language Detection**: Auto-detects and responds in English or Korean

### 🔍 Intelligent Search

#### Adaptive Precision Levels
- **HIGH Precision** (≥70% confidence): Strict specialty filter + semantic ranking
- **MEDIUM Precision** (30-70% confidence): Specialty filter + general fallback
- **LOW Precision** (<30% confidence): Distance-first, all facility types

#### Hybrid Search Strategy
- **Agentic Retrieval Loop**: The LLM chooses dense general search, specific BM25 evidence search, refinement, or finish
- **Specific BM25 Search**: Exact-token retrieval over individual summaries, highlights, amenities, and medical facts
- **Vector Semantic Search**: Understanding intent and context
- **Dynamic Alpha Routing**: Query classifier determines optimal BM25/Vector balance
  - FACTUAL queries (α=0.3-0.5): Favor keyword matching
  - MIXED queries (α=0.6-0.8): Balance both approaches

#### Keyword System
- **Hard Keywords** (MUST-have): parking, MRI, insurance, weekend hours (+2000 boost)
- **Soft Keywords** (preferences): friendly, modern, experienced (+500 boost)
- **Negative Keywords** (avoid): crowded, rushed (-1500 penalty)
- **Negative Hard Keywords** (MUST-NOT): no parking, no English (-5000 penalty)

### 📍 Location Intelligence

#### Multi-Modal Location Support
1. **GPS Coordinates**: Precise distance-based search
2. **District/Neighborhood**: Zone-based filtering (e.g., "Gangnam-gu")
3. **City-wide Search**: All 25 districts without location bias
4. **Address Text**: Fuzzy matching for landmarks and full addresses

#### Geocoding Services
- **Primary**: Google Maps Geocoding API
- **Fallback**: Kakao Maps API (Korea-optimized)
- **Reverse Geocoding**: Convert GPS → District/Dong
- **Address Verification**: Standardization and validation

#### Distance Filtering
- **Travel Preferences**: Walking Distance (0.5 km), Nearby (1 km), Close (2 km), Moderate (5 km), Flexible (10 km), Willing to Travel (15 km), and Anywhere in Seoul (25 km)
- **Adaptive Weighting**: Closer facilities prioritized unless keywords dominate
- **Emergency Mode**: Distance-only ranking for urgent care

### 🏥 Medical Data

#### Dataset Coverage
- **8,484** unique medical facilities across Seoul
- **25** administrative districts (구)
- **320** represented neighborhoods (동)
- **50+** specialty categories (치과, 피부과, 내과, etc.)

#### Facility Information
- **Basic**: Name, address, phone, category, business hours
- **Review Summaries**: AI-generated from Naver reviews (English + Korean)
- **Raw Reviews**: 1,791,749 searchable, nonempty verbatim comments in an on-demand Parquet snapshot. A script-based scan found 1,735,083 Hangul-without-Latin rows, 5,125 Latin-only English-like rows, 20,427 mixed Hangul-and-Latin rows, and 31,114 other rows. These are script groups, not claims about each review's language.

Reproduce the review counts against the local snapshot:

```bash
backend/venv/bin/python backend/tests/profile_review_languages.py
```

Probe the seeded reverse-target case without calling an LLM:

```bash
backend/venv/bin/python backend/tests/probe_reverse_target.py
```

Build the immutable Phase 3 indexes without an embedding API call:

```bash
PYTHONPATH=backend backend/venv/bin/python -m search.indexes.cli publish \
  --root backend/search_indexes \
  --version 2026-09-02-v1

PYTHONPATH=backend backend/venv/bin/python -m search.indexes.cli activate \
  --root backend/search_indexes \
  --version 2026-09-02-v1
```

The current release contains 8,484 facility vectors and 2,060,433 evidence
records. It is immutable, source-hash bound, and loaded read-only. Startup
reports it in `/health`. Phase 4 now uses it for scoped facility BM25, dense,
and evidence retrieval with weighted RRF and one bounded retry.

Exercise the live adapter against the local production release without a
network embedding call:

```bash
backend/venv/bin/python scripts/smoke_phase4_retrieval.py
```

See [`backend/search/PHASE4_ARCHITECTURE.md`](backend/search/PHASE4_ARCHITECTURE.md)
for fusion, retry, fallback, and distance behavior.
- **Key Highlights**: Top 5 notable features extracted from reviews
- **Amenities**: Parking, wheelchair access, elevator, etc.
- **English Support**: Confidence score for English-speaking staff
- **Medical Services**: Equipment, procedures, insurance acceptance

### 🚨 Emergency Features
- **Emergency Detection**: Recognizes urgent medical keywords
- **Nearest Facilities**: Finds 3 closest emergency-capable facilities
- **Categories**: 응급실, 종합병원, 국립병원, 시립병원
- **119 Integration**: Provides emergency number with instructions
- **No Filtering**: Prioritizes speed over preference matching

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  - Chat Interface                                               │
│  - Location Sharing (GPS)                                       │
│  - Travel Distance Widget                                       │
│  - Result Cards                                                 │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP/JSON
┌─────────────────▼───────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                          │
│  - /chat (main endpoint)                                        │
│  - /consent (cookie preferences)                                │
│  - /set_travel_preference                                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│              ROUTER-CONTROLLER ARCHITECTURE                     │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────┐     │
│  │   Router     │───▶│  9 Intent Branches:              │     │
│  │  (LLM-based) │    │  - PROVIDE_INFO                  │     │
│  └──────────────┘    │  - CHANGE_CRITERIA               │     │
│                      │  - CONFIRMATION                  │     │
│                      │  - EMERGENCY                     │     │
│                      │  - NEW_SEARCH                    │     │
│                      │  - CHIT_CHAT                     │     │
│                      │  - HELP_RECOVERY                 │     │
│                      └──────────────────────────────────┘     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  EXTRACTION & VALIDATION                        │
│  - Entity Extraction (LLM: openai/gpt-oss-120b)                 │
│  - Keyword Validation (fuzzy matching)                         │
│  - Intent-based Classification (positive vs negative)          │
│  - Location Verification (Google/Kakao APIs)                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                    SEARCH EXECUTION                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  PATH A: Strict Filtering                            │     │
│  │  - Hard keyword matching (word boundaries)           │     │
│  │  - Negative keyword exclusion                        │     │
│  │  - Specialty filter (if confidence ≥0.5)            │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  PATH B: Additive Scoring (if Path A < 3 results)   │     │
│  │  - Hard keywords: +2000 boost                        │     │
│  │  - Soft keywords: +500 boost                         │     │
│  │  - Negative keywords: -1500 penalty                  │     │
│  │  - Specialty match: +100 * confidence                │     │
│  │  - Distance decay: -10 * normalized_distance         │     │
│  └──────────────────────────────────────────────────────┘     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  HYBRID RAG RANKING                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Query Router (FACTUAL vs MIXED)                     │     │
│  │  - Determines optimal α (BM25/Vector balance)        │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  BM25 Keyword Search                                 │     │
│  │  - Tokenized corpus (8,484 facility documents)                       │     │
│  │  - Okapi BM25 algorithm                              │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Vector Semantic Search                              │     │
│  │  - ChromaDB (persistent storage)                     │     │
│  │  - OpenAI text-embedding-3-small                     │     │
│  │  - Cosine similarity ranking                         │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Hybrid Scoring                                       │     │
│  │  - Normalize BM25 and vector scores                  │     │
│  │  - Combine: α * vector + (1-α) * BM25                │     │
│  │  - Sort by combined score                            │     │
│  └──────────────────────────────────────────────────────┘     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                 RESPONSE GENERATION                             │
│  - Context Building (top N facilities)                         │
│  - LLM Generation (openai/gpt-oss-120b)                         │
│  - Keyword-First Formatting                                    │
│  - English/Korean Response                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Frontend chat interface
2. **Language Detection** → English or Korean
3. **Intent Routing** → LLM classifies into 9 intents
4. **Entity Extraction** → Specialty, location, keywords, travel distance
5. **Validation** → Fuzzy keyword matching, geocoding verification
6. **Search Execution** → Path A (strict) → Path B (scoring) if needed
7. **Hybrid Ranking** → BM25 + Vector with dynamic alpha
8. **Response Generation** → Keyword-focused summaries
9. **Frontend Rendering** → Cards with maps, highlights, navigation

---

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Language**: Python 3.12
- **ASGI Server**: Uvicorn with uvloop

### AI/ML
- **LLM Provider**: OpenRouter (`openai/gpt-oss-120b`)
  - Entity extraction
  - Intent routing
  - Query classification
  - Response generation
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: ChromaDB (persistent)
- **Keyword Search**: BM25Okapi (rank-bm25)

### Data Processing
- **DataFrame**: Pandas + NumPy
- **Geospatial**: Custom haversine implementation
- **Text Processing**: Python re, fuzzy matching

### External APIs
- **Google Maps API**
  - Geocoding
  - Reverse geocoding
  - Place search
  - Place details
- **Kakao Maps API** (fallback)
  - Geocoding
  - Reverse geocoding

### Storage
- **Vector Store**: ChromaDB (local persistent)
- **Dataset**: Parquet files (Hugging Face Hub)

### Development
- **Environment**: python-dotenv
- **Logging**: Python logging module
- **CORS**: FastAPI middleware

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- pip or conda
- Git

### Clone Repository

```bash
git clone https://github.com/ValerianFourel/SeoulDoctor.git
cd SeoulDoctor/backend
```

### Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n seoul-med python=3.12
conda activate seoul-med
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The tested Python dependency versions are pinned in `backend/requirements.txt`.

### Environment Variables

Create `.env` file:

```bash
# OpenRouter API (LLM)
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_PROVIDER=openrouter
OPENROUTER_CHAT_MODEL=openai/gpt-oss-120b
OPENROUTER_AGENT_MODEL=openai/gpt-oss-120b
CHAT_RATE_LIMIT_REQUESTS=12
CHAT_RATE_LIMIT_WINDOW_SECONDS=60

# Optional Groq fallback
GROQ_API_KEY=your_groq_api_key

# OpenAI API (Embeddings)
OPENAI_API_KEY=your_openai_api_key

# Google Maps API
GOOGLE_MAPS_API_KEY=your_google_maps_key

# Kakao Maps API (fallback)
KAKAO_REST_API_KEY=your_kakao_key

# Optional: Naver API (future)
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_secret
```

Create `frontend/.env.local` for the browser application:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SITE_URL=http://localhost:3000

# Optional: omit this until a real AdSense slot has been provisioned
NEXT_PUBLIC_ADSENSE_SLOT_ID=your_adsense_slot_id
```

### Download Dataset

The dataset is automatically downloaded from Hugging Face Hub on first startup. Alternatively, manually download:

```bash
python -c "from utils import download_and_cache_parquet; download_and_cache_parquet()"
```

### Initialize Vector Database

```bash
# The ChromaDB collection is automatically created on first run
# Force recreation (if needed):
python
>>> from rag_pipeline import RAGPipeline
>>> import pandas as pd
>>> df = pd.read_parquet('path/to/data.parquet')
>>> rag = RAGPipeline(...)
>>> rag.initialize_collection(df, force_recreate=True)
```

---

## ⚙️ Configuration

### Server Configuration

**main.py:**
```python
# Defaults
DEFAULT_LAT = 37.5219  # Yeouido, Seoul
DEFAULT_LON = 126.9243
DEFAULT_MAX_DISTANCE = 5.0  # km

# CORS origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://seouldoc.io",
    "https://www.seouldoc.io"
]

# ChromaDB path
CHROMA_PATH = "./chroma_db"
```

### Distance Mapping

**config.py:**
```python
DISTANCE_MAPPING = {
    "Walking Distance": 0.5,
    "Nearby": 1.0,
    "Close": 2.0,
    "Moderate": 5.0,
    "Flexible": 10.0,
    "Willing to Travel": 15.0,
    "Anywhere in Seoul": 25.0,
}
```

### Logging

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce third-party noise
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
```

---

## 🔌 API Documentation

### POST `/chat`

Main conversational endpoint.

**Request:**
```json
{
  "message": "find me a friendly dentist with parking in Gangnam",
  "current_state": {
    "specialty": null,
    "location": null,
    "keywords": [],
    "hard_keywords": [],
    "negative_keywords": [],
    "latitude": null,
    "longitude": null,
    "max_distance_km": 5.0,
    "turn_count": 0
  }
}
```

**Response:**
```json
{
  "response": "I found 4 facilities in Gangnam matching your criteria:\n\n1. **Seoul Dental Clinic** - ✓ Parking, Friendly staff: Dedicated parking lot, warm and welcoming atmosphere...",
  "state": {
    "specialty": "치과",
    "specialty_confidence": 0.95,
    "location": "Gangnam",
    "district": "강남구",
    "keywords": ["friendly"],
    "hard_keywords": ["parking"],
    "latitude": 37.4979,
    "longitude": 127.0276,
    "max_distance_km": 5.0,
    "turn_count": 1,
    "ready_to_search": true,
    "search_executed": true
  },
  "results": [
    {
      "place_id": "12345",
      "name": "Seoul Dental Clinic",
      "category": "치과",
      "address": "강남구 테헤란로 123",
      "distance_km": 0.8,
      "has_english": true,
      "Summaries": ["Patients praise the friendly..."],
      "amenities": {"parking": true, "wheelchair": false}
    }
  ]
}
```

### POST `/consent`

Update cookie consent preferences.

**Request:**
```json
{
  "analytics": true,
  "advertising": false,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Consent preferences updated",
  "consent": {
    "analytics": true,
    "advertising": false,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### POST `/set_travel_preference`

Directly set travel distance from UI widget.

**Request:**
```json
{
  "travel_label": "Far",
  "current_state": { ... }
}
```

**Response:**
```json
{
  "response": "Got it! I'll search within 10km (Far).",
  "state": {
    "travel_label": "Far",
    "max_distance_km": 10.0,
    "travel_confidence": 1.0
  }
}
```

---

## 🧩 Core Components

### 1. Router (`router_prompt.py`)

**Intent Classification System**

9 conversation intents with confidence scoring:

| Intent | Description | Example |
|--------|-------------|---------|
| `PROVIDE_INFO` | User providing search criteria | "dentist in Gangnam" |
| `CHANGE_CRITERIA` | Modifying existing search | "actually, make it Hongdae" |
| `CONFIRMATION` | Confirming to proceed | "yes", "okay", "go ahead" |
| `EMERGENCY` | Urgent medical need | "emergency", "urgent care" |
| `NEW_SEARCH` | Starting over | "reset", "new search" |
| `CHIT_CHAT` | Casual conversation | "thanks!", "how are you?" |
| `HELP_RECOVERY` | Confused/stuck | "I don't understand" |

**Router Prompt Structure:**
```python
ROUTER_PROMPT = """
Classify user intent based on:
- Message content
- Current state (specialty, location, ready_to_search)
- Turn count
- Search status

Return JSON:
{
  "intent": "PROVIDE_INFO",
  "confidence": 0.95,
  "reasoning": "User specified dentist and location"
}
"""
```

### 2. Entity Extraction (`extract_entities()`)

**LLM-based extraction with post-processing:**

```python
def extract_entities(user_message: str, consent) -> Dict:
    """
    Extracts:
    - specialty (치과, 피부과, etc.)
    - location (district, GPS, address)
    - hard_keywords (parking, MRI, etc.)
    - soft_keywords (friendly, modern, etc.)
    - negative_keywords (crowded, rushed, etc.)
    - negative_hard_keywords (no parking, etc.)
    - travel_label (Walking, Moderate, etc.)
    """
```

**Key Features:**
- **Intent-based classification**: "I want unfriendly" → soft_keywords
- **Fallback extraction**: Direct word matching if LLM fails
- **Post-processing correction**: Moves misclassified keywords
- **Geocoding verification**: Validates locations via Google/Kakao

### 3. Keyword Validation (`fuzzy_keyword_match()`)

**Prevents hallucination:**

```python
def fuzzy_keyword_match(keyword: str, message: str) -> bool:
    """
    Returns True if:
    1. Exact substring match
    2. All significant words appear
    3. Synonym match
    
    Examples:
    - "English support" matches "English-speaking staff" ✓
    - "parking lot" matches "parking" ✓
    - "unfriendly" matches "rude" ✓
    - "professional" does NOT match "unprofessional" ✓
    """
```

**Synonym Dictionary:**
```python
synonyms = {
    'parking': ['parking lot', 'car park', '주차'],
    'unfriendly': ['rude', 'impolite', 'brusque', '무례'],
    'english': ['english-speaking', 'english support', '영어'],
    # ... 20+ synonym groups
}
```

### 4. Search Execution (`execute_search()`)

**Adaptive 3-tier precision system:**

```python
# HIGH PRECISION (≥70% confidence)
- Strict specialty filter
- Semantic ranking dominates
- Expand radius if <3 results

# MEDIUM PRECISION (30-70% confidence)  
- Specialty filter + general fallback
- Balance semantic + distance
- Add 10 general facilities if sparse

# LOW PRECISION (<30% confidence)
- Distance-first ranking
- All facility types
- Skip semantic filtering
```

**Dual-path ranking:**
```python
# PATH A: Strict filtering
- Hard keyword AND matching (word boundaries)
- Negative keyword exclusion
- Specialty filter (if high confidence)

# PATH B: Additive scoring (if Path A < 3)
- Hard keywords: +2000
- Soft keywords: +500
- Negative keywords: -1500
- Specialty match: +100 * confidence
- Distance decay: -10 * normalized_distance
```

### 5. RAG Pipeline (`rag_pipeline.py`)

**Hybrid retrieval system:**

```python
class RAGPipeline:
    def hybrid_search(query, alpha=0.7):
        """
        Combines BM25 and vector search
        
        Alpha controls balance:
        - 0.0 = Pure keyword (BM25)
        - 1.0 = Pure semantic (vector)
        
        Query router determines optimal alpha:
        - FACTUAL: α=0.3-0.5 (favor keywords)
        - MIXED: α=0.6-0.8 (balance both)
        """
```

**Components:**
- **BM25 Index**: 40k tokenized documents
- **Vector Store**: ChromaDB with OpenAI embeddings
- **Query Router**: LLM-based FACTUAL vs MIXED classification
- **Score Normalization**: Weighted combination of both methods

### 6. Response Generation (`GENERATION_PROMPT`)

**Keyword-first template:**

```python
GENERATION_PROMPT = """
CRITICAL: Highlight keyword matches in EVERY facility

Format:
1. **Facility Name** - ✓ Keyword1, Keyword2: Description
2. **Facility Name** - ✓ Keyword3: Description

Examples:
✓ Parking, English-speaking: Dedicated lot, fluent staff
✓ Unfriendly, direct: No-nonsense care, efficient visits
✓ Weekend hours, Insurance: Open Sat/Sun, accepts 건강보험
"""
```

---

## 🔍 Search Strategy

### Overview

The search system uses a **3-tier adaptive precision model** that automatically adjusts based on specialty confidence:

```
User Query → Confidence Assessment → Precision Level → Search Strategy
```

### Precision Tiers

#### 🔬 HIGH Precision (≥70% confidence)
**When**: Clear specialty mentioned ("dermatologist", "dentist")

**Strategy:**
1. Strict specialty filtering
2. Semantic ranking dominates (70-95% weight)
3. Expand radius progressively if <3 results (1.5x → 2x → 3x → 5x → 50km)
4. No generic fallback

**Example Query**: "dermatologist in Gangnam"
```
✓ Specialty: 피부과 (0.95 confidence)
✓ Filter: Only 피부과 facilities
✓ Ranking: 85% semantic + 15% distance
✓ Expansion: If <3 results, expand to 15km
```

#### 🎯 MEDIUM Precision (30-70% confidence)
**When**: Ambiguous specialty or generic terms

**Strategy:**
1. Specialty filter applied
2. Add 10 general facilities as fallback
3. Balanced semantic + distance (60-80% semantic)
4. Include nearby alternatives

**Example Query**: "doctor with parking"
```
✓ Specialty: 병원,의원 (0.6 confidence)
✓ Filter: 병원,의원 facilities + 10 general nearby
✓ Ranking: 70% semantic + 30% distance
✓ Disclaimer: "I included some general clinics..."
```

#### 🎲 LOW Precision (<30% confidence)
**When**: Very vague or "any doctor" requests

**Strategy:**
1. NO specialty filtering
2. Distance-first ranking (pure proximity)
3. All facility types eligible
4. Skip semantic ranking (too vague)

**Example Query**: "any doctor" or "아무 의사"
```
✓ Specialty: null (0.2 confidence)
✓ Filter: None (all facilities)
✓ Ranking: 100% distance
✓ Note: General search mode
```

### Dual-Path Ranking

Every search runs **two parallel paths** and merges results:

#### 🛤️ PATH A: Strict Filtering
**Goal**: Find facilities that MUST match criteria

```python
1. Hard keyword AND matching (all must be present)
   Example: "parking" AND "English-speaking" → only facilities with BOTH

2. Negative keyword exclusion
   Example: "not crowded" → exclude any mention of "crowded"

3. Word boundary matching
   ✓ "professional" matches "professional staff"
   ✗ "professional" does NOT match "unprofessional"

4. Specialty filter (if confidence ≥0.5)
```

**Result**: 0-N facilities (strict matches only)

#### 🛤️ PATH B: Additive Scoring
**Goal**: Fill to 3-5 results using weighted scoring

Only runs if Path A returns <3 results.

```python
Base Score = 0

# Positive boosts
+ Hard keywords: +2000 per match
+ Soft keywords: +500 per match
+ Specialty match: +100 * confidence

# Negative penalties
- Negative keywords: -1500 per match
- Negative hard keywords: -5000 per match
- Distance decay: -10 * (distance/max_distance)

Final Score = Base + Boosts - Penalties
```

**Merge Strategy:**
- If Path A ≥3: Use Path A only
- If Path A <3: Path A + top N from Path B (fill to 5)

### Hybrid RAG Ranking

After dual-path filtering, apply semantic ranking:

#### Query Router
```python
FACTUAL Query (α=0.3-0.5):
- "dentist with parking in Gangnam"
- Favor keyword matching
- BM25 weight: 50-70%

MIXED Query (α=0.6-0.8):
- "friendly clinic near me"
- Balance keyword + semantic
- BM25 weight: 20-40%
```

#### Hybrid Score Calculation
```python
# Normalize both scores to 0-1
normalized_bm25 = bm25_score / max_bm25
normalized_vector = 1 - (vector_distance / max_distance)

# Combine with alpha
hybrid_score = (1 - α) * normalized_bm25 + α * normalized_vector

# Sort by hybrid score (descending)
```

#### Combined Ranking
```python
# Adaptive weighting based on radius and confidence
if radius ≤ 2km:
    relevance_weight = 0.70  # 70% semantic + 30% distance
elif radius ≤ 5km:
    relevance_weight = 0.85  # 85% semantic + 15% distance
elif radius > 10km:
    relevance_weight = 0.95  # 95% semantic + 5% distance

# Confidence adjustment
if specialty_confidence < 0.4:
    relevance_weight *= 0.75  # Reduce trust in semantic
elif specialty_confidence < 0.7:
    relevance_weight *= 0.90

# Minimum 50% semantic weight (keywords always matter)
final_weight = max(0.50, relevance_weight)

# Combined score
combined = final_weight * semantic_score + (1 - final_weight) * distance_score
```

### City-Wide Search

Special handling when no specific location:

```python
Triggers:
- "citywide", "anywhere in Seoul", "서울 전체"
- Just "Seoul" or "서울" alone
- No location provided

Behavior:
- NO distance calculation (all 25 districts equal)
- NO distance-based sorting
- Pure semantic ranking
- Results labeled "across Seoul's 25 districts"
```

---

## 🏷️ Keyword System

### Philosophy

**Users know what they want** - even if unconventional. The system treats all keywords equally without moral judgment.

### Keyword Types

#### 🔒 Hard Keywords (MUST-have, factual)
**Boost**: +2000 per match

**Examples:**
- `parking` → Facility MUST have parking lot
- `MRI` → Facility MUST have MRI equipment
- `insurance` → Facility MUST accept insurance
- `weekend hours` → Facility MUST be open weekends
- `English-speaking` → Facility MUST have English support

**Matching**:
```python
# Word boundary matching (prevents false positives)
"parking" matches:
  ✓ "parking lot available"
  ✓ "Free parking for patients"
  ✗ "spark-ing new facility" (different word)

# Synonyms
"parking" also matches:
  ✓ "car park"
  ✓ "주차장" (Korean)
  ✓ "garage"
```

#### 💭 Soft Keywords (preferences, subjective)
**Boost**: +500 per match

**Examples:**
- `friendly` → Prefer warm, welcoming staff
- `modern` → Prefer contemporary facilities
- `experienced` → Prefer veteran doctors
- `clean` → Prefer hygienic environment

**Unconventional but VALID:**
- `unfriendly` → Prefer direct, no-nonsense style
- `rude` → Prefer blunt, efficient communication
- `expensive` → Prefer premium, high-end options
- `rushed` → Prefer quick, time-efficient visits
- `cold` → Prefer impersonal, clinical approach

**Intent Detection:**
```python
"I want an unfriendly doctor" 
→ soft_keywords: ["unfriendly"] ✓

"avoid unfriendly doctors"
→ negative_keywords: ["unfriendly"] ✓
```

#### 🚫 Negative Keywords (avoid, subjective)
**Penalty**: -1500 per match

**Examples:**
- `crowded` → Avoid busy, packed facilities
- `rushed` → Avoid hurried consultations
- `unfriendly` → Avoid cold staff
- `dirty` → Avoid poor hygiene

**Triggers:**
- "avoid X"
- "not X"
- "don't want X"
- "skip X"

#### ⛔ Negative Hard Keywords (MUST-NOT, factual)
**Penalty**: -5000 per match

**Examples:**
- `no parking` → Facility MUST NOT lack parking
- `no weekend hours` → Facility MUST NOT be weekday-only
- `without elevator` → Facility MUST NOT lack elevator

**Triggers:**
- "without X"
- "no X"
- "excluding X"

### Extraction Process

```python
1. LLM Extraction (intent-based)
   ↓
2. Post-processing Correction
   - Check for "I want/need" → positive
   - Check for "avoid/not" → negative
   - Move misclassified unconventional keywords
   ↓
3. Fallback Extraction (if LLM fails)
   - Direct word matching
   - Quality adjective detection
   ↓
4. Fuzzy Validation
   - Remove hallucinated keywords
   - Synonym matching
   - Word boundary verification
```

### Examples

#### Example 1: Conventional Keywords
```
Input: "friendly dentist with parking in Gangnam"

Extraction:
✓ Specialty: 치과 (0.95 confidence)
✓ Hard Keywords: ["parking"]
✓ Soft Keywords: ["friendly"]
✓ Location: Gangnam

Search:
- PATH A: Filter for facilities with "parking"
- Boost +500 for "friendly" mentions in reviews
- Semantic ranking prioritizes both
```

#### Example 2: Unconventional Keywords
```
Input: "I need an unfriendly doctor who rushes appointments"

Extraction:
✓ Specialty: 병원,의원 (0.7 confidence)
✓ Soft Keywords: ["unfriendly", "rushes appointments"]
✓ Intent: POSITIVE (user wants these qualities)

Post-processing:
- LLM might put "unfriendly" in negative_keywords
- Correction detects "I need" → moves to soft_keywords

Search:
- Boost +500 for facilities described as:
  * "direct", "no-nonsense", "efficient"
  * "quick consultations", "fast service"
  * "minimal small talk", "time-focused"

Generation:
"✓ Direct, efficient: Quick consultations, no-nonsense approach"
```

#### Example 3: Mixed Positive/Negative
```
Input: "avoid crowded clinics, want modern equipment"

Extraction:
✓ Soft Keywords: ["modern equipment"]
✓ Negative Keywords: ["crowded"]

Search:
- Boost +500 for "modern", "contemporary", "advanced"
- Penalty -1500 for "crowded", "busy", "packed"
- Balance: Modern but not crowded
```

#### Example 4: Hard Exclusion
```
Input: "hospital without weekend hours, clean and professional"

Extraction:
✓ Soft Keywords: ["clean", "professional"]
✓ Negative Hard Keywords: ["weekend hours"]

Search:
- Penalty -5000 for facilities with weekend/Saturday/Sunday hours
- Boost +500 for "clean" and "professional"
- Result: Weekday-only, high-quality facilities
```

---

## 🤖 Agentic RAG Pipeline

### Active Architecture

```
User query + bilingual facets
              │
              ▼
┌──────────────────────────────────────────────────┐
│ Function-calling agent: openai/gpt-oss-120b      │
│ 10 tool iterations by default; hard cap of 12    │
└──────────────┬───────────────────────────────────┘
               │
   ┌───────────┼──────────────────┐
   ▼           ▼                  ▼
Facility    Indexed evidence   Multilingual comments
semantic    BM25               DuckDB + Parquet
search      summaries/facts    1.8M verbatim rows
   │           │                  │
   └───────────┴─────────┬────────┘
                         ▼
        LLM reads original reviews, selects evidence,
        and returns faithful translations + reasons
                         │
                         ▼
       RRF/distance ranking + provenance-aware UI
```

The OpenRouter model uses real local function calls: `search_facilities`,
`search_indexed_evidence`, `search_multilingual_comments`,
`select_comment_evidence`, and `finish_search`. Raw review access is limited
to facility IDs in the active specialty and location candidate scope. Reviewer names are never sent
to the model, and every translation remains attached to its original-language
text, evidence ID, facility, and visit date. Retrieved text is always treated as
untrusted evidence. Set `ENABLE_RAW_REVIEWS=false` to run in meta-review-only
mode.

### Document Indexing

**Text Blob Construction:**
```python
# For each facility
text_blob = f"{name} ({category}). {summary_en} {summary_kr} {highlights}"

# Example:
"Seoul Dental Clinic (치과). Patients praise friendly staff and 
modern equipment. 친절한 직원과 현대적 장비. parking, English support, 
weekend hours"
```

**Specific BM25 Indexing:**
```python
# Each summary/highlight/fact becomes its own evidence record
records = build_specific_evidence_records(facilities)

# Literal tokenizer: no stemming and no substring matching
specific_corpus = [tokenize_exact(record["text"]) for record in records]

specific_bm25 = BM25Okapi(specific_corpus)
```

**Vector Indexing:**
```python
# Generate embedding
embedding = openai.embeddings.create(
    model="text-embedding-3-small",
    input=text_blob
)

# Store in ChromaDB
vector_db.add(
    ids=[place_id],
    documents=[text_blob],
    embeddings=[embedding],
    metadatas=[{
        "category": category,
        "district": district,
        "has_english": has_english
    }]
)
```

### Legacy Hybrid Fallback

The earlier FACTUAL/MIXED alpha router remains available as a fallback through
`semantic_search()`, but production ranking now uses `agentic_search()`.

**Router Prompt:**
```python
QUERY_ROUTER_PROMPT = """
Classify query type:

FACTUAL: Specific factual requirements
- Keywords: parking, MRI, insurance, weekend, English
- Characteristics: Verifiable, objective, binary
- Examples: "dentist with parking", "MRI near Gangnam"
→ Suggested α: 0.3-0.5 (favor keyword matching)

MIXED: Subjective preferences + factual
- Keywords: friendly, modern, experienced, clean
- Characteristics: Semantic, qualitative, nuanced
- Examples: "friendly dentist", "modern clinic"
→ Suggested α: 0.6-0.8 (balance both methods)

Return JSON:
{
  "intent": "FACTUAL" or "MIXED",
  "suggested_alpha": 0.4,
  "reasoning": "Query asks for parking (factual)"
}
```

**Alpha Calculation:**
```python
def calculate_alpha(route_decision, manual_mode=None):
    suggested_alpha = route_decision['suggested_alpha']
    
    # Manual override
    if manual_mode == "FACTUAL_ONLY":
        return max(0.3, min(suggested_alpha, 0.5))
    elif manual_mode == "MIXED":
        return max(0.6, min(suggested_alpha, 0.8))
    
    # Auto mode with clipping (0.3-1.0)
    return max(0.3, min(suggested_alpha, 1.0))
```

### Hybrid Search

**BM25 Search:**
```python
def bm25_search(query_text, n_results=200):
    # Tokenize query
    tokenized_query = query_text.lower().split()
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # Top N results
    top_indices = np.argsort(scores)[::-1][:n_results]
    
    return {
        "ids": [bm25_doc_ids[i] for i in top_indices],
        "scores": [scores[i] for i in top_indices]
    }
```

**Vector Search:**
```python
def vector_search(query_text, n_results=200):
    # Generate query embedding
    results = vector_db.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    return {
        "ids": results['ids'][0],
        "distances": results['distances'][0]
    }
```

**Hybrid Combination:**
```python
def hybrid_search(query_text, alpha=0.7, n_results=200):
    # Get both results
    bm25_results = bm25_search(query_text, n_results)
    vector_results = vector_search(query_text, n_results)
    
    # Normalize BM25 scores (0-1)
    max_bm25 = max(bm25_results['scores'])
    normalized_bm25 = {
        doc_id: score / max_bm25
        for doc_id, score in zip(bm25_results['ids'], bm25_results['scores'])
    }
    
    # Convert vector distances to similarities (0-1)
    max_distance = max(vector_results['distances'])
    normalized_vector = {
        doc_id: 1 - (distance / max_distance)
        for doc_id, distance in zip(vector_results['ids'], vector_results['distances'])
    }
    
    # Combine scores
    combined_scores = {}
    for doc_id in set(normalized_bm25.keys()) | set(normalized_vector.keys()):
        bm25_score = normalized_bm25.get(doc_id, 0)
        vector_score = normalized_vector.get(doc_id, 0)
        
        combined_scores[doc_id] = (1 - alpha) * bm25_score + alpha * vector_score
    
    # Sort by combined score
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n_results]
    
    return {
        "ids": [doc_id for doc_id, _ in sorted_results],
        "scores": [score for _, score in sorted_results],
        "method": "hybrid",
        "alpha": alpha
    }
```

### Context Building

**For LLM Generation:**
```python
def build_context_for_llm(df_subset, n_results=10, language="English"):
    context_parts = []
    
    for _, row in df_subset.head(n_results).iterrows():
        facility_info = [
            f"Name: {row['name']}",
            f"Category: {row['category']}",
            f"District: {row['file_district']}",
            f"Distance: {row['distance_km']:.1f}km"
        ]
        
        # Add summary (language-specific)
        if language == "Korean":
            summary = row['Summaries_Korean'][0]
        else:
            summary = row['Summaries'][0]
        facility_info.append(f"Summary: {summary}")
        
        # Add highlights
        highlights = [h['topic'] for h in row['Key_Highlights'][:5]]
        facility_info.append(f"Highlights: {', '.join(highlights)}")
        
        context_parts.append("\n".join(facility_info))
    
    return "\n\n---\n\n".join(context_parts)
```

---

## 📊 Data Pipeline

### Source Data

**Naver Maps Web Scraping:**
- 8,484 medical facilities in the current Seoul snapshot
- Categories: 50+ specialties (치과, 피부과, 내과, etc.)
- Reviews: Millions of user reviews
- Metadata: Amenities, hours, contact info, GPS

### Review Processing

**AI Summarization:**
```python
# For each facility
reviews = fetch_naver_reviews(place_id)

# Generate English summary
summary_en = llm.summarize(
    reviews,
    language="English",
    focus="patient experience, staff attitude, facility quality"
)

# Generate Korean summary
summary_kr = llm.summarize(
    reviews,
    language="Korean",
    focus="환자 경험, 직원 태도, 시설 품질"
)

# Extract key highlights
highlights = llm.extract_highlights(
    reviews,
    count=5,
    format="topic + percentage"
)
```

**Example Output:**
```json
{
  "Summaries": [
    "Patients consistently praise the clinic's modern equipment and friendly staff. The doctor takes time to explain procedures thoroughly. Parking can be difficult during peak hours."
  ],
  "Summaries_Korean": [
    "환자들은 현대적인 장비와 친절한 직원을 칭찬합니다. 의사가 시술을 자세히 설명합니다. 피크 시간대에 주차가 어려울 수 있습니다."
  ],
  "Key_Highlights": [
    {"topic": "Modern equipment", "percentage": 45.2},
    {"topic": "Friendly staff", "percentage": 38.7},
    {"topic": "Thorough explanations", "percentage": 32.1},
    {"topic": "Parking difficulties", "percentage": 15.6},
    {"topic": "Short wait times", "percentage": 12.3}
  ]
}
```

### Data Schema

**Parquet File Structure:**
```python
{
  "place_id": str,           # Unique identifier
  "name": str,               # Facility name
  "category": str,           # 치과, 피부과, etc.
  "address": str,            # Full address
  "lat": float,              # Latitude
  "lon": float,              # Longitude
  "file_district": str,      # 강남구, 마포구, etc.
  "file_dong": str,          # 역삼동, 홍대동, etc.
  "phone": str,              # Phone number
  "business_hours": str,     # Operating hours
  "website": str,            # Official website
  
  # Review-derived
  "Summaries": list[str],           # English summaries
  "Summaries_Korean": list[str],    # Korean summaries
  "Key_Highlights": list[dict],     # Top 5 topics
  
  # Amenities (parsed from reviews + metadata)
  "amenities": {
    "parking": bool,
    "wheelchair_accessible": bool,
    "elevator": bool,
    "weekend_hours": bool
  },
  
  # Medical info
  "medical_info_parsed": {
    "equipment": list[str],     # MRI, CT, X-ray, etc.
    "procedures": list[str],    # Colonoscopy, etc.
    "insurance": bool           # Accepts insurance
  },
  
  # English support
  "has_english": bool,
  "english_confidence_score": float
}
```

### Data Updates

**Incremental Updates:**
```bash
# Weekly update script
python scripts/update_data.py --incremental

# Full rebuild (monthly)
python scripts/update_data.py --full-rebuild

# Vector DB re-indexing
python scripts/rebuild_vectors.py --force
```

---

## 💬 Conversation Flow

### State Machine

```
┌─────────────┐
│   INITIAL   │
│  (greeting) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  GATHERING  │◄────────────┐
│(collecting  │             │
│ criteria)   │             │
└──────┬──────┘             │
       │                    │
       │ (criteria          │ (need more
       │  complete)         │  info)
       ▼                    │
┌─────────────┐             │
│  SEARCHING  │─────────────┘
│ (executing  │
│  search)    │
└──────┬──────┘
       │
       │ (results
       │  returned)
       ▼
┌─────────────┐
│  COMPLETE   │
│ (showing    │
│  results)   │
└──────┬──────┘
       │
       │ (refine or
       │  new search)
       ▼
   (back to GATHERING)
```

### Example Conversation

```
User: "Hi"
Bot: "Hello! Tell me what kind of doctor you need, and I'll find options..."
State: { phase: "gathering", turn: 0 }

User: "dentist"
Bot: "Got it! Where should I search?"
State: { specialty: "치과", phase: "gathering", turn: 1 }

User: "Gangnam"
Bot: "I found 5 dentists in Gangnam: 1. Seoul Dental..."
State: { specialty: "치과", district: "강남구", phase: "complete", turn: 2 }

User: "actually make it with parking"
Bot: "I found 3 dentists in Gangnam with parking: 1. Modern Dental..."
State: { specialty: "치과", district: "강남구", hard_keywords: ["parking"], turn: 3 }

User: "thanks!"
Bot: "You're welcome! Feel free to search again anytime."
State: { phase: "complete", turn: 4 }
```

### State Transitions

| Current Phase | User Input | Next Phase |
|---------------|------------|------------|
| gathering | Provides specialty | gathering (if no location) |
| gathering | Provides location | gathering (if no specialty) |
| gathering | Provides both | searching |
| gathering | "yes"/"okay" | searching (if ready) |
| searching | Results returned | complete |
| complete | "change to X" | gathering → searching |
| complete | "reset" | initial |
| any | "emergency" | emergency |

---

## 🛠️ Development

### Running Locally

```bash
# Activate environment
source venv/bin/activate

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access API
curl http://localhost:8000/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "dentist in Gangnam", "current_state": {}}'
```

### Testing

**Manual Testing:**
```bash
# Test extraction
python
>>> from main import extract_entities
>>> result = extract_entities("unfriendly doctor with parking")
>>> print(result)

# Test search
python
>>> from main import execute_search
>>> from models import State
>>> state = State(specialty="치과", district="강남구")
>>> response, results = execute_search(state, "dentist")
>>> print(response)
```

**Test Cases:**
```python
# test_keywords.py
test_cases = [
    {
        "input": "I need an unfriendly doctor",
        "expected": {
            "soft_keywords": ["unfriendly"],
            "negative_keywords": []
        }
    },
    {
        "input": "avoid crowded clinics",
        "expected": {
            "soft_keywords": [],
            "negative_keywords": ["crowded"]
        }
    },
    {
        "input": "parking and English-speaking",
        "expected": {
            "hard_keywords": ["parking", "English-speaking"],
            "soft_keywords": []
        }
    }
]
```

### Debugging

**Enable verbose logging:**
```python
# main.py
logging.basicConfig(level=logging.DEBUG)

# See extraction details
logger.debug(f"RAW EXTRACTION: {extracted}")
logger.debug(f"AFTER VALIDATION: {validated}")
logger.debug(f"PATH A RESULTS: {len(path_a_results_df)}")
logger.debug(f"PATH B RESULTS: {len(path_b_results_df)}")
```

**Common Issues:**

1. **No results returned**: Check specialty confidence
   ```python
   logger.info(f"Specialty: {state.specialty} (conf={state.specialty_confidence})")
   ```

2. **Keywords not matching**: Check fuzzy matching
   ```python
   logger.info(f"Keyword '{kw}' validation: {fuzzy_keyword_match(kw, message)}")
   ```

3. **Wrong intent**: Check router output
   ```python
   logger.info(f"Router intent: {intent} (reasoning: {route['reasoning']})")
   ```

### Performance Optimization

**BM25 Index:**
- Pre-tokenized corpus (8,484 facility documents)
- Search time: ~50-100ms

**Vector Search:**
- ChromaDB persistent storage (no rebuilding)
- Query time: ~200-500ms (200 results)

**Hybrid Search:**
- Combined: ~300-600ms
- Caching: Response memoization for common queries

**LLM Calls:**
- OpenRouter latency depends on the selected model and provider
- Batching: Process multiple extractions in parallel (future)

---

## 📁 Project Structure

```
seoul-doctor-matchmaker/
├── backend/
│   ├── main.py                  # FastAPI app, routes, search logic
│   ├── agentic_retrieval.py     # Exact tokens, evidence chunks, plan validation
│   ├── rag_pipeline.py          # Agentic dense/BM25 retrieval loop
│   ├── prompt.py                # All LLM prompts
│   ├── models.py                # Pydantic state models
│   ├── utils.py                 # Helper functions
│   ├── distance.py              # Haversine, fuzzy matching
│   ├── location.py              # Google/Kakao geocoding
│   ├── cookies.py               # GDPR consent handling
│   ├── deterministic.py         # Static responses
│   ├── .env                     # API keys (not committed)
│   ├── requirements.txt         # Python dependencies
│   └── chroma_db/               # Persistent vector store
│
├── frontend/                    # React app (separate repo)
│
├── data/
│   └── seoul_facilities.parquet # Main dataset
│
├── scripts/
│   ├── update_data.py           # Data refresh script
│   ├── rebuild_vectors.py       # Vector DB rebuild
│   └── test_extraction.py       # Testing utilities
│
├── docs/
│   ├── API.md                   # API documentation
│   ├── ARCHITECTURE.md          # System design
│   └── PROMPTS.md               # Prompt engineering guide
│
├── README.md                    # This file
└── LICENSE
```

---

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Keep functions under 50 lines

### Testing
- Add test cases for new features
- Ensure all existing tests pass
- Test with both English and Korean inputs

---

## 🙏 Acknowledgments

- **OpenRouter**: LLM routing for `openai/gpt-oss-120b`
- **OpenAI**: High-quality embeddings (text-embedding-3-small)
- **ChromaDB**: Vector database
- **Google Maps**: Geocoding services
- **Kakao Maps**: Korea-optimized location services
- **Hugging Face**: Dataset hosting

---

## 📧 Contact

**Project Maintainer**: [@ValerianFourel](https://github.com/ValerianFourel)
- Email: seouldoc.io@gmail.com
- GitHub: [@ValerianFourel](https://github.com/ValerianFourel)
- Website: https://seouldoc.io

---

## 🔮 Future Enhancements

### Short-term
- [ ] Add more Korean language support in prompts
- [ ] Implement caching for common queries
- [ ] Add user authentication
- [ ] Save favorite facilities

### Medium-term
- [ ] Real-time availability checking
- [ ] Appointment booking integration
- [ ] Multi-city support (Busan, Incheon)
- [ ] Mobile app (iOS/Android)

### Long-term
- [ ] Telemedicine integration
- [ ] Insurance verification
- [ ] Medical record integration
- [ ] AI symptom checker

---

**Built with ❤️ for the international community in Seoul**
