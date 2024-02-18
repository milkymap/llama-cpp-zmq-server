import asyncio

import json 
import httpx 

def handler():
    async def _main():
        async with httpx.AsyncClient() as client:

            async with client.stream(method='POST', url='http://localhost:8000/models/download_model', json={
                "url2model": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q2_K.gguf?download=true",
                'name_of_model': 'phi-2.Q2_K_001.gguf',
                'bytes_monitor_rate': 65536 * 2
                }) as resp:
                async for line in resp.aiter_lines():
                    stream_data = json.loads(line)
                    print(stream_data)

    asyncio.run(main=_main())

if __name__ == '__main__':
    handler()