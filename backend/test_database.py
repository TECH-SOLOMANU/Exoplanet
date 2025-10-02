import asyncio
from app.services.nasa_service import NASADataService
from app.core.database import get_database

async def test_database():
    db = get_database()
    count = await db.exoplanets.count_documents({})
    print(f'Total NASA records in database: {count}')
    
    # Sample a few records to see the data quality
    sample = []
    async for doc in db.exoplanets.find().limit(3):
        sample.append({
            'name': doc.get('pl_name'),
            'status': doc.get('pl_status'),
            'source': doc.get('source'),
            'period': doc.get('pl_orbper'),
            'radius': doc.get('pl_rade')
        })
    
    print(f'\nSample records:')
    for i, record in enumerate(sample, 1):
        print(f'{i}. {record}')

if __name__ == "__main__":
    asyncio.run(test_database())