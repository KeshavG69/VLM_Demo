from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.db.mongo import MongoDb
from agno.run.agent import RunEvent, RunOutput, RunOutputEvent
from dotenv import load_dotenv
import os
import logging
from typing import AsyncGenerator, Dict, Any
import json
import asyncio
from contextlib import suppress
import time

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Image Query API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MongoDB instance
mongodb_instance = None

def get_mongodb_instance():
    """Get or create MongoDB instance"""
    global mongodb_instance
    if mongodb_instance is None:
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            try:
                mongodb_instance = MongoDb(
                    db_url=mongo_uri,
                    db_name="es-qastage01"
                )
                logger.info("MongoDB connected successfully")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                mongodb_instance = None
        else:
            logger.warning("MONGO_URI not found in environment")
    return mongodb_instance


def extract_text(content: Any) -> str:
    """Extract text from content"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content and isinstance(content["content"], str):
            return content["content"]
    return str(content)


async def stream_image_agent_response(
    query: str,
    agent: Agent,
    image_bytes: bytes
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream agent response with image"""

    # Yield initial event
    yield {
        "event": "QueryAnalysing",
        "content": "Analyzing image and your query..."
    }

    start_time = time.monotonic()
    first_delta_emitted = False

    try:
        # Create Image object
        agno_image = Image(content=image_bytes)

        # Run agent with image
        response_stream = agent.arun(
            query,
            stream=True,
            stream_intermediate_steps=True,
            images=[agno_image]
        )

        async for run_chunk in response_stream:
            payload: Dict[str, Any]

            if isinstance(run_chunk, RunOutputEvent):
                payload = run_chunk.to_dict()
            elif isinstance(run_chunk, RunOutput):
                payload = run_chunk.to_dict()
                payload.setdefault("event", RunEvent.run_completed.value)
            else:
                payload = {
                    "event": getattr(run_chunk, "event", RunEvent.run_content.value),
                    "content": str(run_chunk),
                }

            agno_event = payload.get("event", RunEvent.run_content.value)

            if agno_event == RunEvent.run_started.value:
                yield {
                    "event": "run.started",
                    "run_id": payload.get("run_id"),
                    "session_id": payload.get("session_id"),
                }
                continue

            if agno_event in {RunEvent.run_content.value, RunEvent.run_intermediate_content.value}:
                delta_text = extract_text(payload.get("content"))
                if delta_text:
                    if not first_delta_emitted:
                        first_delta_emitted = True
                        ttft_ms = (time.monotonic() - start_time) * 1000.0
                        logger.info(f"TTFT: {ttft_ms:.1f}ms")
                    yield {
                        "event": "message.delta",
                        "content": delta_text,
                    }
                continue

            if agno_event == RunEvent.run_completed.value:
                yield {
                    "event": "run.completed",
                    "run_id": payload.get("run_id"),
                }

                final_text = extract_text(payload.get("content"))
                if final_text:
                    yield {
                        "event": "message.completed",
                        "content": final_text,
                    }
                continue

            if agno_event == RunEvent.run_error.value:
                yield {
                    "event": "error",
                    "error": payload.get("content") or payload.get("message"),
                }
                continue

    except Exception as exc:
        logger.error(f"Streaming error: {exc}", exc_info=True)
        yield {
            "event": "error",
            "error": str(exc),
        }


async def create_sse_event_stream(
    events: AsyncGenerator[Dict[str, Any], None]
) -> AsyncGenerator[str, None]:
    """Convert events to SSE format"""

    iterator = events.__aiter__()
    pending = asyncio.create_task(iterator.__anext__())
    event_id = 0

    try:
        while True:
            try:
                event = await asyncio.wait_for(pending, timeout=30)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            except StopAsyncIteration:
                yield "data: [DONE]\n\n"
                break
            except Exception as exc:
                event_id += 1
                fallback = {
                    "event": "error",
                    "error": str(exc),
                }
                data = json.dumps(fallback)
                yield f"id: {event_id}\nevent: error\ndata: {data}\n\n"
                yield "data: [DONE]\n\n"
                break
            else:
                event_id += 1
                event_name = event.get("event", "message.delta")
                payload = {k: v for k, v in event.items() if k != "event"}
                data = json.dumps(payload, ensure_ascii=False, default=str)
                yield f"id: {event_id}\nevent: {event_name}\ndata: {data}\n\n"
                pending = asyncio.create_task(iterator.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending


@app.post("/image-query")
async def image_query(
    query: str = Form(...),
    session_id: str = Form(...),
    image: UploadFile = File(...)
):
    """Image query endpoint with streaming response"""

    try:
        # Validate inputs
        if not query or query.strip() == "":
            raise HTTPException(status_code=400, detail="Query is required")
        if not session_id or session_id.strip() == "":
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Read image bytes
        image_bytes = await image.read()

        # Get MongoDB instance
        db_instance = get_mongodb_instance()

        # Create agent
        agent = Agent(
            name="Image Analysis Agent",
            model=OpenAIChat(
                id="gpt-4.1-nano",
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            session_id=session_id,
            markdown=True,
            stream_intermediate_steps=True,
            db=db_instance,
            add_history_to_context=True,
            num_history_runs=5,
            enable_agentic_memory=True,
            enable_user_memories=True,
            telemetry=False,
            debug_mode=True,
            description="AI assistant that analyzes images and answers questions",
            instructions=[
                "You are an expert image analysis assistant.",
                "Analyze images carefully and provide detailed, accurate descriptions.",
                "Answer user questions about the image content precisely.",
                "When the user asks follow-up questions, reference your previous analysis.",
                "Be concise but thorough in your responses."
            ]
        )

        # Create SSE stream
        async def sse_stream():
            events = stream_image_agent_response(query, agent, image_bytes)
            async for chunk in create_sse_event_stream(events):
                yield chunk

        # Return streaming response
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers=headers
        )

    except Exception as e:
        logger.error(f"Error in image query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mongodb_connected": get_mongodb_instance() is not None
    }

@app.get("/")
async def health():
    """Health check endpoint"""
    return {
        "status": "Eveything Working Fine",
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
