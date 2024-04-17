from typing import Optional

import openai
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, status, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models.openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel

from io_processing import *
from logger import logger
from utils import is_url, is_base64

gpt_model = get_config_value("llm", "gpt_model", None)

emotion_classifier_prompt = get_config_value("llm", "emotion_classifier_prompt", None)
welcome_msg_classifier_prompt = get_config_value("llm", "welcome_msg_classifier_prompt", None)
feedback_msg_classifier_prompt = get_config_value("llm", "feedback_msg_classifier_prompt", None)

llm_client = openai.OpenAI()

app = FastAPI(title="ALL BOT Service",
              #   docs_url=None,  # Swagger UI: disable it by setting docs_url=None
              redoc_url=None,  # ReDoc : disable it by setting docs_url=None
              swagger_ui_parameters={"defaultModelsExpandDepth": -1},
              description='',
              version="1.0.0"
              )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_host = get_config_value('redis', 'redis_host', None)
redis_port = get_config_value('redis', 'redis_port', None)
redis_index = get_config_value('redis', 'redis_index', None)

# Connect to Redis
redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_index)  # Adjust host and port if needed

language_code_list = get_config_value('request', 'supported_lang_codes', None).split(",")
if language_code_list is None:
    raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

learning_language_list = get_config_value('request', 'learn_language', None)
if learning_language_list is None:
    raise HTTPException(status_code=422, detail="learn_language not configured!")

welcome_msg = json.loads(get_config_value("response_messages", "welcome_message", None))
welcome_greeting_resp_msg = json.loads(get_config_value("response_messages", "welcome_greeting_response_message", None))
welcome_other_resp_msg = json.loads(get_config_value("response_messages", "welcome_other_response_message", None))
get_user_feedback_msg = json.loads(get_config_value("response_messages", "get_user_feedback_message", None))
feedback_positive_resp_msg = json.loads(get_config_value("response_messages", "feedback_positive_response_message", None))
feedback_other_resp_msg = json.loads(get_config_value("response_messages", "feedback_other_response_message", None))


# Define a function to store and retrieve data in Redis
def store_data(key, value):
    redis_client.set(key, value)


def retrieve_data(key):
    data_from_redis = redis_client.get(key)
    return data_from_redis.decode('utf-8') if data_from_redis is not None else None


@app.on_event("startup")
async def startup_event():
    logger.info('Invoking startup_event')
    load_dotenv()
    logger.info('startup_event : Engine created')


@app.on_event("shutdown")
async def shutdown_event():
    logger.info('Invoking shutdown_event')
    logger.info('shutdown_event : Engine closed')


class OutputResponse(BaseModel):
    audio: str = None
    text: str = None
    content_id: str = None


class ConversationResponse(BaseModel):
    audio: str = None


class ResponseForQuery(BaseModel):
    output: ConversationResponse


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"


class GetFeedBackRequest(BaseModel):
    user_id: str = None


class WelcomeUserInputModel(BaseModel):
    user_id: str = None
    conversation_language: str = None
    learning_language: str = None
    audio: str = None


class WelcomeUserResponseInputModel(BaseModel):
    user_id: str = None
    audio: str = None


class GetContentRequest(BaseModel):
    user_id: str = None
    language: str = None


class UserAnswerRequest(BaseModel):
    user_id: str = None
    content_id: str = None
    audio: str = None
    language: str = None
    original_text: str = None


class GetContentResponse(BaseModel):
    conversation: Optional[ConversationResponse]
    content: Optional[OutputResponse]


def _handle_error(error: ToolException) -> str:
    return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool."
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to ALL BOT Service"}


@app.get(
    "/health",
    tags=["Health Check"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    include_in_schema=True
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


llm = ChatOpenAI(model=gpt_model, temperature=1)

load_dotenv()


def invoke_llm(user_id: str, user_statement: str, prompt: str, session_id: str, language: str) -> str:
    logger.debug({"intent_classifier": "classifier_prompt", "user_id": user_id, "language": language, "session_id": session_id, "user_statement": user_statement})
    res = llm_client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_statement}
        ],
    )
    message = res.choices[0].message.model_dump()
    llm_response = message["content"]
    logger.info({"intent_classifier": "openai_response", "user_id": user_id, "language": language, "session_id": session_id, "response": llm_response})

    return llm_response


def emotions_classifier(user_id: str, user_statement: str, session_id: str, language: str) -> str:
    logger.info({"user_id": user_id, "language": language, "session_id": session_id, "user_statement": user_statement})
    user_session_emotions = retrieve_data(user_id + "_" + language + "_" + session_id + "_emotions")
    logger.info({"user_id": user_id, "language": language, "session_id": session_id, "user_session_emotions": user_session_emotions})

    emotion_category = invoke_llm(user_id, user_statement, emotion_classifier_prompt, session_id, language)

    if user_session_emotions:
        user_session_emotions = json.loads(user_session_emotions)
        user_session_emotions.append(emotion_category)
    else:
        user_session_emotions = [emotion_category]

    store_data(user_id + "_" + language + "_" + session_id + "_emotions", json.dumps(user_session_emotions))
    return emotion_category


def validate_user(user_id: str):
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid user_id input!")

    user_learning_language = retrieve_data(user_id + "_learning_language")
    user_conversation_language = retrieve_data(user_id + "_conversation_language")
    user_session_id = retrieve_data(user_id + "_" + user_learning_language + "_session")

    if user_session_id is None or user_learning_language is None or user_conversation_language is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="User session not found!")

    return user_session_id, user_learning_language, user_conversation_language

@app.post("/v1/welcome_user", include_in_schema=True)
async def welcome_user_msg(request: WelcomeUserInputModel, x_session_id: str = Header(None, alias="X-Request-ID")) -> ConversationResponse:
    conversation_language = request.conversation_language.strip().lower()
    if conversation_language is None or conversation_language == "" or conversation_language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported conversation language code entered!")

    learning_language = request.learning_language.strip().lower()
    if learning_language is None or learning_language == "" or learning_language not in learning_language_list:
        raise HTTPException(status_code=422, detail="Unsupported learning language code entered!")

    user_id = request.user_id
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid user_id input!")

    store_data(user_id + "_learning_language", learning_language)
    store_data(user_id + "_conversation_language", conversation_language)
    store_data(user_id + "_" + learning_language + "_session", x_session_id)
    logger.info({"user_id": user_id, "conversation_language": conversation_language, "learning_language": learning_language, "current_session_id": x_session_id})

    return_welcome_msg = welcome_msg[conversation_language]
    logger.info({"user_id": user_id, "x_session_id": x_session_id, "return_welcome_msg": return_welcome_msg})
    return ConversationResponse(audio=return_welcome_msg)


@app.post("/v1/welcome_user_response", include_in_schema=True)
async def welcome_user_resp_msg(request: WelcomeUserResponseInputModel) -> ConversationResponse:
    user_id = request.user_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_id)
    audio = request.audio
    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")

    user_statement_reg, user_statement, error_message = process_incoming_voice(audio, user_conversation_language)
    logger.info({"user_id": user_id, "audio_converted_eng_text:": user_statement})
    # classify welcome_user_resp emotion into ['Excited', 'Happy', 'Curious', 'Bored', 'Confused', 'Angry', 'Sad']
    emotions_classifier(user_id, user_statement, user_session_id, user_learning_language)
    # classify welcome_user_resp intent into 'greeting' and 'other'
    user_intent = invoke_llm(user_id, user_statement, welcome_msg_classifier_prompt, user_session_id, user_learning_language)
    # Based on the intent, return response
    if user_intent == "greeting":
        return_welcome_intent_msg = welcome_greeting_resp_msg[user_conversation_language]
    else:
        return_welcome_intent_msg = welcome_other_resp_msg[user_conversation_language]

    logger.info({"user_id": user_id, "x_session_id": user_session_id, "return_welcome_intent_msg": return_welcome_intent_msg})
    return ConversationResponse(audio=return_welcome_intent_msg)


@app.post("/v1/get_feedback_msg", include_in_schema=True)
async def query_feedback(request: GetFeedBackRequest) -> ConversationResponse:
    user_id = request.user_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_id)
    get_user_feedback_message = get_user_feedback_msg[user_conversation_language]

    return ConversationResponse(audio=get_user_feedback_message)



@app.post("/v1/feedback_user_response", include_in_schema=True)
async def feedback_user_resp_msg(request: WelcomeUserResponseInputModel) -> ConversationResponse:
    user_id = request.user_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_id)
    audio = request.audio
    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")

    user_statement_reg, user_statement, error_message = process_incoming_voice(audio, user_conversation_language)
    logger.info({"user_id": user_id, "audio_converted_eng_text:": user_statement})
    # classify feedback_user_resp emotion into ['Excited', 'Happy', 'Curious', 'Bored', 'Confused', 'Angry', 'Sad']
    emotions_classifier(user_id, user_statement, user_session_id, user_learning_language)

    # classify feedback_user_resp intent into 'positive', 'other'
    user_intent = invoke_llm(user_id, user_statement, feedback_msg_classifier_prompt, user_session_id, user_learning_language)
    # Based on the intent, return response
    if user_intent == "positive":
        return_feedback_intent_msg = feedback_positive_resp_msg[user_conversation_language]
    else:
        return_feedback_intent_msg = feedback_other_resp_msg[user_conversation_language]

    logger.info({"user_id": user_id, "x_session_id": user_session_id, "return_feedback_intent_msg": return_feedback_intent_msg})
    return ConversationResponse(audio=return_feedback_intent_msg)
    ###

