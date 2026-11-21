
import asyncio
import sys

async def test():
    try:
        from app.utils.config import settings
        from app.utils.logger import configure_logging
        configure_logging(settings.log_level)
        print('Config OK:', settings.agriparam_device)
        
        from app.services.qdrant_service import qdrant_service
        qdrant_service.connect()
        print('Qdrant OK')
        
        from app.rag.embedder import embedding_service
        embedding_service.load()
        print('Embedder OK')
        
        from app.llm.agriparam import agriparam_service
        agriparam_service.load()
        print('LLM OK')
        
    except Exception as e:
        import traceback
        print('CRASH AT:', e)
        traceback.print_exc()

asyncio.run(test())
