import asyncio
import os
import json
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType, PayloadSchemaType
import asyncpg
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://gitglobe:gitglobe@localhost:5433/gitglobe"
    qdrant_url: str = "http://localhost:6333"

    class Config:
        env_file = "../../pipeline/.env"

async def main():
    settings = Settings()
    print("Connecting to Postgres...")
    conn = await asyncpg.connect(settings.database_url)
    
    print("Connecting to Qdrant...")
    qclient = AsyncQdrantClient(url=settings.qdrant_url)
    
    collection_name = "gitglobe_repos"
    
    # Setup collection
    collections = await qclient.get_collections()
    if any(c.name == collection_name for c in collections.collections):
        print(f"Collection {collection_name} already exists. Recreating...")
        await qclient.delete_collection(collection_name)
    
    # 768 is typical for Vertex AI or Voyage
    # But wait! We need to know what dimension the embeddings are.
    row = await conn.fetchrow("SELECT embedding_dim FROM repo WHERE embedding IS NOT NULL LIMIT 1")
    if not row:
        print("No embeddings found in Postgres!")
        return
        
    dim = row["embedding_dim"]
    print(f"Detected embedding dimension: {dim}")
    
    await qclient.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(default_segment_number=2),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )
    
    print("Creating payload indexes...")
    await qclient.create_payload_index(collection_name, "language", PayloadSchemaType.KEYWORD)
    await qclient.create_payload_index(collection_name, "domain", PayloadSchemaType.KEYWORD)
    await qclient.create_payload_index(collection_name, "stars", PayloadSchemaType.INTEGER)
    
    print("Fetching repos from Postgres...")
    
    # Need to decode the embedding bytes into a list of floats.
    import struct
    
    query = """
    SELECT id, full_name, description, language, domain, stars, embedding 
    FROM repo 
    WHERE embedding IS NOT NULL
    """
    
    batch_size = 500
    points = []
    
    async with conn.transaction():
        async for row in conn.cursor(query):
            raw_bytes = row["embedding"]
            # embeddings are typically stored as a flat array of 32-bit floats
            num_floats = len(raw_bytes) // 4
            vector = list(struct.unpack(f"{num_floats}f", raw_bytes))
            
            payload = {
                "full_name": row["full_name"],
                "description": row["description"],
                "language": row["language"],
                "domain": row["domain"],
                "stars": row["stars"]
            }
            
            points.append(PointStruct(id=row["id"], vector=vector, payload=payload))
            
            if len(points) >= batch_size:
                await qclient.upsert(collection_name=collection_name, points=points)
                print(f"Upserted {len(points)} points...")
                points = []
                
    if points:
        await qclient.upsert(collection_name=collection_name, points=points)
        print(f"Upserted final {len(points)} points.")
        
    print("Done seeding Qdrant!")

if __name__ == "__main__":
    asyncio.run(main())
