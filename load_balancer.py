from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import yaml
from pathlib import Path
import time
from fastapi.background import BackgroundTasks
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional, TypeVar, Generic
from pydantic import BaseModel
import json
import sglang_router
from multiprocessing import Process

app = FastAPI()
routers = {} # key: model_name, value: Router   
current_endpoints = {}

T = TypeVar('T')
class Router(BaseModel, Generic[T]):
    model_name: str
    router: T
    process: Optional[T] = None
    port: int
    host: Optional[str] = "127.0.0.1"
    created_at: Optional[int] = int(time.time())


# Example endpoints_config.yaml:
# anthropic/claude-2: http://localhost:8001
# anthropic/claude-instant-1: http://localhost:8002
# openai/gpt-4: http://localhost:8003
# openai/gpt-3.5-turbo: http://localhost:8004
# Configuration file path
# Note: This file should be in the same directory as the load_balancer.py script

CONFIG_FILE = "endpoints_config.yaml"
CHECK_INTERVAL = 60  # seconds
# Port range for SGLang router instances
MIN_PORT = 30000
MAX_PORT = 50000
PORT_SHIFT = 2
SGLANG_PORT = MIN_PORT

async def load_config():
    """Load endpoints configuration from YAML file and update router if changed"""
    global routers, current_endpoints, SGLANG_PORT
    try:
        print("Starting load_config...")  # Debug log
        if Path(CONFIG_FILE).exists():
            print("Reading config file...")  # Debug log
            with open(CONFIG_FILE, 'r') as f:
                new_endpoints = yaml.safe_load(f)
                
            if not isinstance(new_endpoints, dict):
                print("Error: Config file must contain a dictionary")
                return
                
            if new_endpoints != current_endpoints:
                print("Config changed, stopping old routers...")  # Debug log
                # Stop existing routers
                for router in routers.values():
                    if hasattr(router, 'process'):
                        print(f"Stopping router process for {router.model_name}")  # Debug log
                        router.process.terminate()
                        router.process.join(timeout=5)  # Add timeout
                        if router.process.is_alive():
                            print(f"Warning: Process for {router.model_name} didn't stop cleanly")
                            router.process.kill()  # Force kill if needed

                routers = {}    
                current_endpoints = new_endpoints

                # Initialize new routers
                for model_name, urls in new_endpoints.items():
                    print(f"Starting router for {model_name}, {urls}, {SGLANG_PORT}")  # Debug log
                    if isinstance(urls, str):
                        urls = [urls]
                    
                    # Validate URLs start with http://
                    for url in urls:
                        if not url.startswith('http://'):
                            print(f"Error: URL {url} must start with http://")
                            return
                    try:
                        router = Router(
                            model_name=model_name,
                            port=SGLANG_PORT,
                            host="127.0.0.1",
                            router=sglang_router.Router(
                                worker_urls=urls,
                                policy=sglang_router.PolicyType.CacheAware,
                                host="127.0.0.1",
                                port=SGLANG_PORT,
                                verbose=True
                            )
                        )
                        
                        # Start router in a separate process with timeout
                        process = Process(target=router.router.start, daemon=True)  # Make it daemon
                        print(f"Starting process for {model_name}")  # Debug log
                        process.start()
                        router.process = process
                        
                        routers[model_name] = router
                        SGLANG_PORT += PORT_SHIFT
                        if SGLANG_PORT > MAX_PORT:
                            print("Error: No available ports for SGLang router instances")
                            return
                            
                    except Exception as e:
                        print(f"Error starting router for {model_name}: {e}")
                        continue
                
                print(f"Router update complete")  # Debug log
                
    except Exception as e:
        print(f"Error in load_config: {e}")
        import traceback
        traceback.print_exc()
    print(f"Current routers: {routers}")

async def periodic_config_check():
    """Periodically check and reload config file"""
    while True:
        await load_config()
        await asyncio.sleep(CHECK_INTERVAL)

@app.on_event("startup")
async def startup_event():
    """Load initial config and start periodic checking"""
    await load_config()
    asyncio.create_task(periodic_config_check())

"""
Load Balancer for ML Model Inference

This script implements a FastAPI-based load balancer that distributes requests across multiple ML model endpoints.
It supports both POST requests for specific model inference and GET requests that aggregate results from all models.
"""

async def check_model(req: Request, routers: dict, suffix_used: str, headers: dict):
    """Handle POST requests for specific model inference, with streaming support."""
    try:
        data = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse request JSON: {str(e)}")
    
    model_name = data.get("model", None)
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name not provided")
    
    print("check_model", routers)
    router = routers.get(model_name, None)
    if not router:
        raise HTTPException(status_code=400, detail="Router not found for model")
    
    full_url = f"http://{router.host}:{router.port}/v1/{suffix_used}"
    headers["Content-Type"] = "application/json"
    
    # If streaming is requested, handle it differently
    if data.get("stream", False):
        async def stream_generator():
            async with aiohttp.ClientSession() as session:
                async with session.post(full_url, json=data) as response:
                    async for line in response.content:
                        if line:
                            yield line
        
        return stream_generator(), 200
    
    # Non-streaming request handling
    async with aiohttp.ClientSession() as session:
        async with session.post(full_url, json=data) as response:
            print("response", response)
            try:
                return await response.json(), response.status
            except aiohttp.client_exceptions.ContentTypeError:
                # If Content-Type is missing, try to parse the response text as JSON
                text = await response.text()
                try:
                    return json.loads(text), response.status
                except json.JSONDecodeError:
                    raise HTTPException(status_code=500, 
                                     detail="Invalid JSON response from upstream server")

async def merge_across_models(req: Request, routers: dict, suffix_used: str, headers: dict):
    """
    Handle GET requests by aggregating responses from all model endpoints.
    
    Args:
        req (Request): FastAPI request object containing the client's request data
        avail_endpoints (dict): Dictionary mapping model names to their endpoint URLs
        suffix_used (str): API endpoint suffix (path after /v1/)
        headers (dict): HTTP headers to forward to the model endpoints
    
    Returns:
        tuple: (merged_response_json, status_code) combining results from all endpoints
    """
    final_response = None
    final_status_code = None
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        router = routers.get(model_name, None)
        if not router:
            raise HTTPException(status_code=400, detail="Router not found for model")
        full_url = f"http://{router.host}:{router.port}/v1/{suffix_used}"
        tasks.append(session.get(full_url, params=req.query_params, headers=headers))
        
        responses = await asyncio.gather(*tasks)
        for response in responses:
            response_json = await response.json()
            if final_response is None:
                final_response = response_json
            else:
                final_response["data"].extend(response_json["data"])
            final_status_code = final_status_code or response.status
            
    return final_response, final_status_code

async def list_models():
    """Return list of available models in OpenAI format"""
    models = []
    for model_name, router in routers.items():
        models.append({
            "id": model_name,
            "object": "model",
            "created": router.created_at,  # Use the stored creation time
            "owned_by": "organization-owner"
        })
    
    return {
        "object": "list",
        "data": models
    }

@app.api_route("/v1/{suffix:path}", methods=["POST", "GET"])
async def api_router(suffix: str, request: Request):
    """Main API route handler with streaming support."""
    headers = dict(request.headers)
    excluded_headers = ['Host', 'Content-Length', 'Content-Type']
    headers = {key: value for key, value in headers.items() if key not in excluded_headers}

    if request.method == "POST":
        try:
            response_data, response_status_code = await check_model(request, routers, suffix, headers)
        except Exception as e:
            print(f"Error in check_model: {e}")
            return JSONResponse(
                content={
                    "error": str(e),
                    "solution": "follow the openai api spec"
                },
                status_code=500
            )
        
        # Handle streaming response
        if isinstance(response_data, AsyncGenerator):
            return StreamingResponse(
                response_data,
                media_type='text/event-stream',
                status_code=response_status_code
            )
        
        return JSONResponse(content=response_data, status_code=response_status_code)
    else:  # GET
        if suffix == "models":
            return JSONResponse(content=await list_models(), status_code=200)
        try:    
            response_json, response_status_code = await merge_across_models(request, routers, suffix, headers)
        except Exception as e:
            print(f"Error in merge_across_models: {e}")
            return JSONResponse(
                content={
                    "error": str(e),
                    "solution": "follow the openai api spec"
                },
                status_code=500
            )
        return JSONResponse(content=response_json, status_code=response_status_code)


if __name__ == "__main__":
    """
    Entry point for running the FastAPI application.
    Starts the uvicorn server on host 0.0.0.0 and port 8000.
    
    Usage:
        python load_balancer.py
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)