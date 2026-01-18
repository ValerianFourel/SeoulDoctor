import os
import json
import pandas as pd
import numpy as np
import requests
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from groq import Groq
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from specialties import SPECIALTY_CATEGORY_MAP
from distance import haversine, fuzzy_match_location
from models import ChatRequest,State
from utils import safe_convert_to_python
from prompt import EXTRACTION_PROMPT, GENERATION_PROMPT
# --- 1. CONFIGURATION ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HF_TOKEN = os.getenv("HF_TOKEN") 
HF_REPO_ID = "ValerianFourel/seoul-medical-facilities"
HF_FILENAME = "facilities_metareviews_rag_ready.parquet"

CHROMA_PATH = "./chroma_db"
LOCAL_PARQUET_PATH = "./local_facilities_cache.parquet"

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243
vector_db = None
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)


def download_and_cache_parquet():
    """Download parquet from HuggingFace and cache locally."""
    if os.path.exists(LOCAL_PARQUET_PATH):
        print(f"✅ Loading cached parquet from {LOCAL_PARQUET_PATH}")
        df = pd.read_parquet(LOCAL_PARQUET_PATH)
    else:
        print(f"📥 Downloading parquet from HuggingFace...")
        remote_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=HF_FILENAME, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        df = pd.read_parquet(remote_path)
        
        # Cache locally
        df.to_parquet(LOCAL_PARQUET_PATH)
        print(f"💾 Cached parquet locally to {LOCAL_PARQUET_PATH}")
    
    return df



def build_context_for_llm(df_subset: pd.DataFrame, n_results: int = 10) -> str:
    """Build rich context from filtered parquet data for LLM."""
    context_parts = []
    
    for idx, row in df_subset.head(n_results).iterrows():
        facility_info = []
        
        facility_info.append(f"Name: {row['name']}")
        facility_info.append(f"Category: {row['category']}")
        
        if 'file_district' in row.index and pd.notna(row['file_district']):
            facility_info.append(f"District: {row['file_district']}")
        if 'file_dong' in row.index and pd.notna(row['file_dong']):
            facility_info.append(f"Neighborhood: {row['file_dong']}")
        
        if 'distance_km' in row.index and pd.notna(row['distance_km']) and row['distance_km'] > 0:
            facility_info.append(f"Distance: {row['distance_km']:.1f}km away")
        
        # Handle Summaries (ndarray)
        if 'Summaries' in row.index:
            summaries = row['Summaries']
            if isinstance(summaries, (list, np.ndarray)) and len(summaries) > 0:
                facility_info.append(f"Summary (EN): {summaries[0]}")
        
        # Handle Summaries_Korean (ndarray)
        if 'Summaries_Korean' in row.index:
            summaries_kr = row['Summaries_Korean']
            if isinstance(summaries_kr, (list, np.ndarray)) and len(summaries_kr) > 0:
                facility_info.append(f"Summary (KR): {summaries_kr[0]}")
        
        # Handle Key_Highlights (ndarray)
        if 'Key_Highlights' in row.index:
            highlights_data = row['Key_Highlights']
            if isinstance(highlights_data, (list, np.ndarray)) and len(highlights_data) > 0:
                topics = []
                for h in highlights_data[:5]:
                    if isinstance(h, dict) and 'topic_en' in h:
                        topics.append(h['topic_en'])
                if topics:
                    facility_info.append(f"Highlights: {', '.join(topics)}")
        
        if 'has_english' in row.index and row['has_english']:
            facility_info.append(f"English Speaking: Yes")
        
        context_parts.append("\n".join(facility_info))
    
    return "\n\n---\n\n".join(context_parts)

# --- 3. LIFESPAN (STARTUP) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities, df_filtered
    print("🚀 Booting SeoulMedBot Backend...")

    try:
        df_facilities = download_and_cache_parquet()
        print(f"✅ Loaded {len(df_facilities)} facilities from parquet")
        
        # FILTER: Only use facilities with Summaries
        df_filtered = df_facilities[df_facilities['Summaries'].notna()].copy()
        print(f"✅ Filtered to {len(df_filtered)} facilities with summaries")
        
        # Show location data
        if 'file_district' in df_filtered.columns:
            unique_districts = df_filtered['file_district'].dropna().unique()
            print(f"📍 Districts ({len(unique_districts)}): {sorted(unique_districts)[:5]}...")
        
        # Check for GPS coordinates - NOTE: columns are 'lat' and 'lon' not 'latitude'/'longitude'
        if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns:
            coords_count = df_filtered[
                (df_filtered['lat'].notna()) & 
                (df_filtered['lon'].notna())
            ].shape[0]
            print(f"🌐 GPS coordinates available for {coords_count}/{len(df_filtered)} facilities")
            if coords_count == 0:
                print(f"   ⚠️ All lat/lon are null - will use text-based location matching only")
        else:
            print(f"⚠️ No lat/lon columns - will use text-based location matching only")
        
        df_filtered['place_id'] = df_filtered['place_id'].fillna('').astype(str)        
    except Exception as e:
        print(f"❌ PARQUET LOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        df_facilities = pd.DataFrame()
        df_filtered = pd.DataFrame()

    # Setup RAG
    client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    try:
        vector_db = client_chroma.get_collection("seoul_med_v3", embedding_function=openai_ef)
        print("✅ Embeddings collection found.")
    except:
        print("⚠️ Creating new embeddings from filtered dataset...")
        vector_db = client_chroma.create_collection("seoul_med_v3", embedding_function=openai_ef)
        
        ids, docs, metas = [], [], []
        for _, row in df_filtered.iterrows():
            summary_en = ""
            summary_kr = ""
            highlights = ""
            
            if isinstance(row['Summaries'], (list, np.ndarray)) and len(row['Summaries']) > 0:
                summary_en = str(row['Summaries'][0])
            if isinstance(row['Summaries_Korean'], (list, np.ndarray)) and len(row['Summaries_Korean']) > 0:
                summary_kr = str(row['Summaries_Korean'][0])
            if isinstance(row['Key_Highlights'], (list, np.ndarray)) and len(row['Key_Highlights']) > 0:
                highlights = ", ".join([h.get('topic', '') for h in row['Key_Highlights'] if isinstance(h, dict)])
            
            text_blob = f"{row['name']} ({row['category']}). {summary_en} {summary_kr} {highlights}"
            
            ids.append(str(row['place_id']))
            docs.append(text_blob)
            metas.append({
                "category": row['category'],
                "district": str(row.get('file_district', '')),
                "has_english": bool(row.get('has_english', False))
            })

        # Batch insert
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            vector_db.add(ids=ids[i:end], documents=docs[i:end], metadatas=metas[i:end])
            
        print("✅ Embeddings indexed.")

    yield
    print("🛑 Shutting down.")

app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- 5. CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    
    # STEP 1: EXTRACT INTENT
    msgs = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"State: {req.current_state.model_dump_json()}\nUser: {req.message}"}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=msgs,
            temperature=0.0,
            max_completion_tokens=1024,
            response_format={"type": "json_object"}
        )
        ai_data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"response": "System error.", "state": req.current_state}

    new_state = ai_data.get("state", {})
    response_text = ai_data.get("response_text", "")
    
    # Fallback language detection
    if not new_state.get("language_pref") or new_state.get("language_pref") == "Korean is fine":
        user_message = req.message.lower()
        has_english = any(char.isalpha() and ord(char) < 128 for char in user_message)
        has_korean = any(0xAC00 <= ord(char) <= 0xD7A3 for char in req.message)
        
        if has_english and not has_korean:
            new_state["language_pref"] = "English Preferred"
        elif has_korean:
            new_state["language_pref"] = "Korean"
        else:
            new_state["language_pref"] = "Korean"
    
    print(f"\n--- 🧠 STATE: {new_state.get('specialty')} in {new_state.get('location')} ---\n")

    # STEP 2: SEARCH
    location_context = ""
    results = []
    
    if new_state.get("ready_to_search"):
        working_df = df_filtered.copy()
        print(f"📊 Starting with {len(working_df)} facilities")
        
        # LOCATION FILTERING - Only text-based (lat/lon are empty)
        if new_state.get("location"):
            print(f"📍 Text-based location search")
            working_df = fuzzy_match_location(new_state['location'], working_df)
            location_context = f"in {new_state['location']}"
            working_df['distance_km'] = 0
        else:
            print(f"📍 City-wide search")
            location_context = "across Seoul"
            working_df['distance_km'] = 0
        
        # CATEGORY FILTER
        korean_category = None
        if new_state.get("specialty"):
            specialty_lower = new_state['specialty'].lower().strip()
            if specialty_lower in SPECIALTY_CATEGORY_MAP:
                korean_category = SPECIALTY_CATEGORY_MAP[specialty_lower]
                print(f"🏥 Specialty '{new_state['specialty']}' → '{korean_category}'")
        
        if korean_category and 'category' in working_df.columns:
            before_count = len(working_df)
            working_df = working_df[working_df['category'].str.contains(korean_category, na=False, case=False)]
            print(f"   ✅ Category filter: {before_count} → {len(working_df)}")
        
        # SEMANTIC RANKING (RAG)
        query_text = f"{new_state.get('specialty', '')} {' '.join(new_state.get('keywords', []))}"
        
        if len(query_text.strip()) > 2 and len(working_df) > 0:
            try:
                print(f"🔍 RAG ranking for: '{query_text}'")
                n_rag_results = min(200, len(vector_db.get()['ids']))
                rag_results = vector_db.query(query_texts=[query_text], n_results=n_rag_results)
                
                if rag_results and 'ids' in rag_results and len(rag_results['ids']) > 0:
                    rag_ranking = {place_id: idx for idx, place_id in enumerate(rag_results['ids'][0])}
                    working_df['relevance_rank'] = working_df['place_id'].apply(lambda pid: rag_ranking.get(pid, 9999))
                    working_df = working_df.sort_values('relevance_rank')
                    print(f"   ✅ RAG ranked {(working_df['relevance_rank'] < 9999).sum()}/{len(working_df)}")
                else:
                    working_df['relevance_rank'] = 9999
            except Exception as e:
                print(f"   ⚠️ RAG error: {e}")
                working_df['relevance_rank'] = 9999
        else:
            working_df['relevance_rank'] = 9999
        
        # ENGLISH FILTER
        language_pref = new_state.get("language_pref", "Korean")
        if "English" in language_pref and 'has_english' in working_df.columns:
            before = len(working_df)
            working_df = working_df[working_df['has_english'] == True]
            print(f"🌐 English filter: {before} → {len(working_df)}")
        
        # GENERATE RESPONSE
        if len(working_df) > 0:
            print(f"✅ Building context from {len(working_df)} facilities")
            facilities_context = build_context_for_llm(working_df, n_results=10)
            
            language = "English" if "English" in language_pref else "Korean"
            
            gen_messages = [{
                "role": "system",
                "content": GENERATION_PROMPT.format(
                    user_query=req.message,
                    location_context=location_context,
                    language=language,
                    facilities_context=facilities_context
                )
            }]
            
            gen_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=gen_messages,
                temperature=1.0,
                max_completion_tokens=1024
            )
            response_text = gen_completion.choices[0].message.content
            
            # Prepare results - USE SAFE CONVERSION FOR ALL FIELDS
            for _, row in working_df.head(3).iterrows():
                result = {
                    "place_id": safe_convert_to_python(row['place_id']),
                    "name": safe_convert_to_python(row['name']),
                    "category": safe_convert_to_python(row['category']),
                }
                
                # Simple fields
                simple_fields = ['address', 'phone', 'business_hours', 'english_confidence_score']
                for field in simple_fields:
                    if field in row.index and pd.notna(row[field]):
                        result[field] = safe_convert_to_python(row[field])
                
                # Location fields
                if 'file_district' in row.index and pd.notna(row['file_district']):
                    result['district'] = safe_convert_to_python(row['file_district'])
                if 'file_dong' in row.index and pd.notna(row['file_dong']):
                    result['dong'] = safe_convert_to_python(row['file_dong'])
                
                # Website
                if 'website' in row.index and pd.notna(row['website']):
                    result['website'] = safe_convert_to_python(row['website'])
                elif 'url' in row.index and pd.notna(row['url']):
                    result['website'] = safe_convert_to_python(row['url'])
                
                # Complex fields - CRITICAL: use safe_convert_to_python for ndarrays
                if 'Summaries' in row.index and isinstance(row['Summaries'], (list, np.ndarray)):
                    result['Summaries'] = safe_convert_to_python(row['Summaries'])
                
                if 'Summaries_Korean' in row.index and isinstance(row['Summaries_Korean'], (list, np.ndarray)):
                    result['Summaries_Korean'] = safe_convert_to_python(row['Summaries_Korean'])
                
                if 'Key_Highlights' in row.index and isinstance(row['Key_Highlights'], (list, np.ndarray)):
                    result['Key_Highlights'] = safe_convert_to_python(row['Key_Highlights'])
                
                if 'amenities' in row.index:
                    result['amenities'] = safe_convert_to_python(row['amenities'])
                
                if 'medical_info_parsed' in row.index and isinstance(row['medical_info_parsed'], dict):
                    result['medical_info_parsed'] = safe_convert_to_python(row['medical_info_parsed'])
                
                # Booleans and numbers
                result['has_english'] = safe_convert_to_python(row.get('has_english', False))
                result['distance_km'] = safe_convert_to_python(row.get('distance_km', 0))
                result['relevance_rank'] = safe_convert_to_python(row.get('relevance_rank', 9999))
                
                results.append(result)
        else:
            language_pref = new_state.get("language_pref", "Korean")
            if "English" in language_pref:
                response_text = "I couldn't find any facilities matching your criteria. Would you like to try a different specialty or area?"
            else:
                response_text = "검색 조건에 맞는 시설을 찾을 수 없습니다. 다른 전문 분야나 지역을 시도해 보시겠어요?"

    return {
        "response": response_text,
        "state": new_state,
        "results": results
    }