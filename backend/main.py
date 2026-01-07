from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import hashlib
from pathlib import Path
import tempfile
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import jwt
import bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AI Chat Pro API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GROQ_API_KEY = ""  # 👈 PASTE YOUR API KEY HERE
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "ai_chat_pro"
JWT_SECRET = "your-secret-key-change-this-in-production"  # Change this!

# MongoDB client
mongo_client = None
db = None

# In-memory storage for vectorstores
vectorstores = {}

# Pydantic models
class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_email: str
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    model: Optional[str] = "llama-3.3-70b-versatile"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    use_pdf: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    tokens: int
    model: str

@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB connection"""
    global mongo_client, db
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL)
        db = mongo_client[DATABASE_NAME]
        await mongo_client.admin.command('ping')
        logger.info(f"✅ Connected to MongoDB: {DATABASE_NAME}")
        
        # Create indexes
        await db.messages.create_index([("session_id", 1), ("timestamp", 1)])
        await db.sessions.create_index([("session_id", 1)], unique=True)
        await db.users.create_index([("email", 1)], unique=True)
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close MongoDB connection"""
    if mongo_client:
        mongo_client.close()

# Helper functions
def create_token(email: str) -> str:
    """Create JWT token"""
    return jwt.encode({"email": email}, JWT_SECRET, algorithm="HS256")

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("email")
    except:
        return None

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current user from token"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization.replace("Bearer ", "")
    email = verify_token(token)
    
    if not email or db is None:
        return None
    
    user = await db.users.find_one({"email": email})
    return user

async def save_message_to_db(session_id: str, role: str, content: str, 
                             user_email: str, tokens: int = 0, model: str = None):
    """Save message to MongoDB"""
    if db is None:
        return
    
    try:
        message_doc = {
            "session_id": session_id,
            "user_email": user_email,
            "role": role,
            "content": content,
            "tokens": tokens,
            "model": model,
            "timestamp": datetime.utcnow()
        }
        
        await db.messages.insert_one(message_doc)
        
        # Update session
        session_update = {
            "$set": {
                "session_id": session_id,
                "user_email": user_email,
                "last_updated": datetime.utcnow()
            },
            "$inc": {
                "message_count": 1,
                "total_tokens": tokens
            },
            "$setOnInsert": {
                "created_at": datetime.utcnow()
            }
        }
        
        await db.sessions.update_one(
            {"session_id": session_id},
            session_update,
            upsert=True
        )
        
    except Exception as e:
        logger.error(f"Error saving message: {e}")

async def get_session_history(session_id: str, limit: int = 10) -> List[dict]:
    """Get recent messages"""
    if db is None:
        return []
    
    try:
        cursor = db.messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit)
        
        messages = []
        async for doc in cursor:
            messages.append({
                "role": doc["role"],
                "content": doc["content"],
                "tokens": doc.get("tokens", 0)
            })
        
        return messages
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []

# ==================== API ROUTES ====================

@app.get("/api", tags=["Info"])
async def api_info():
    """API Information"""
    db_status = "connected" if db is not None else "disconnected"
    return {
        "message": "AI Chat Pro API v2.0",
        "database": db_status,
        "features": ["Authentication", "MongoDB", "PDF Chat", "Multi-user"]
    }

@app.post("/api/auth/signup", tags=["Authentication"])
async def signup(user: UserSignup):
    """User signup"""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Check if user exists
        existing = await db.users.find_one({"email": user.email})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
        
        # Create user
        user_doc = {
            "name": user.name,
            "email": user.email,
            "password": hashed.decode(),
            "created_at": datetime.utcnow()
        }
        
        await db.users.insert_one(user_doc)
        
        # Create token
        token = create_token(user.email)
        
        return {
            "message": "Account created successfully",
            "token": token,
            "user": {
                "name": user.name,
                "email": user.email
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")

@app.post("/api/auth/login", tags=["Authentication"])
async def login(user: UserLogin):
    """User login"""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Find user
        db_user = await db.users.find_one({"email": user.email})
        
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not bcrypt.checkpw(user.password.encode(), db_user["password"].encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
        token = create_token(user.email)
        
        return {
            "message": "Login successful",
            "token": token,
            "user": {
                "name": db_user["name"],
                "email": db_user["email"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Main chat endpoint"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        if not GROQ_API_KEY or GROQ_API_KEY == "":
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # Get session history
        session_history = await get_session_history(request.session_id, limit=10)
        
        # Initialize LLM
        llm = ChatGroq(
            model=request.model,
            groq_api_key=GROQ_API_KEY,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Build messages
        messages = [SystemMessage(content=request.system_prompt)]
        
        # Add PDF context if available
        if request.use_pdf and request.session_id in vectorstores:
            retriever = vectorstores[request.session_id].as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            docs = retriever.invoke(request.message)
            
            if docs:
                context = "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
                messages.append(SystemMessage(
                    content=f"IMPORTANT: You have access to PDF content. Use it to answer:\n\n{context}"
                ))
        
        # Add conversation history
        for msg in session_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        messages.append(HumanMessage(content=request.message))
        
        # Get response
        response = llm.invoke(messages)
        response_text = response.content
        
        # Estimate tokens
        tokens = int(len(response_text.split()) * 1.3)
        
        # Save to database
        await save_message_to_db(request.session_id, "user", request.message, 
                                request.user_email, 0, request.model)
        await save_message_to_db(request.session_id, "assistant", response_text, 
                                request.user_email, tokens, request.model)
        
        return ChatResponse(
            response=response_text,
            tokens=tokens,
            model=request.model
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf", tags=["PDF"])
async def upload_pdf(session_id: str, file: UploadFile = File(...), 
                     current_user: dict = Depends(get_current_user)):
    """Upload and process PDF"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load and process
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(docs, embeddings)
        
        # Store vectorstore
        vectorstores[session_id] = vectorstore
        
        # Save PDF info
        if db is not None:
            await db.pdfs.insert_one({
                "session_id": session_id,
                "user_email": current_user["email"],
                "filename": file.filename,
                "chunk_count": len(docs),
                "uploaded_at": datetime.utcnow()
            })
        
        # Cleanup
        Path(tmp_path).unlink()
        
        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "chunks": len(docs)
        }
        
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-session", tags=["Session"])
async def clear_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Clear session"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        if db is not None:
            await db.messages.delete_many({"session_id": session_id})
            await db.sessions.delete_one({"session_id": session_id})
            await db.pdfs.delete_many({"session_id": session_id})
        
        if session_id in vectorstores:
            del vectorstores[session_id]
        
        return {"message": "Session cleared"}
        
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/sessions", tags=["Session"])
async def get_user_sessions(current_user: dict = Depends(get_current_user)):
    """Get all user sessions"""
    if not current_user or db is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        sessions = []
        cursor = db.sessions.find(
            {"user_email": current_user["email"]}
        ).sort("last_updated", -1).limit(20)
        
        async for doc in cursor:
            sessions.append({
                "session_id": doc["session_id"],
                "message_count": doc.get("message_count", 0),
                "total_tokens": doc.get("total_tokens", 0),
                "last_updated": doc.get("last_updated").isoformat()
            })
        
        return {"sessions": sessions}
        
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", tags=["Info"])
async def get_stats():
    """Get system stats"""
    if db is None:
        return {"error": "Database not available"}
    
    try:
        total_users = await db.users.count_documents({})
        total_sessions = await db.sessions.count_documents({})
        total_messages = await db.messages.count_documents({})
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_messages": total_messages
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== SERVE FRONTEND ====================

# Mount static files for CSS, JS, images
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/", include_in_schema=False)
async def serve_login():
    """Serve login page at root"""
    return FileResponse("../frontend/login.html")

@app.get("/chat", include_in_schema=False)
async def serve_chat():
    """Serve chat page"""
    return FileResponse("../frontend/index.html")

@app.get("/login.html", include_in_schema=False)
async def serve_login_alt():
    """Alternative login route"""
    return FileResponse("../frontend/login.html")

@app.get("/index.html", include_in_schema=False)
async def serve_chat_alt():
    """Alternative chat route"""
    return FileResponse("../frontend/index.html")

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)