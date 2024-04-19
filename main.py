import secrets
import string

import openai
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.pydantic_v1 import BaseModel

from io_processing import *
from logger import logger
from utils import is_url, is_base64

gpt_model = get_config_value("llm", "gpt_model", None)

emotion_classifier_prompt = get_config_value("llm", "emotion_classifier_prompt", None)
welcome_msg_classifier_prompt = get_config_value("llm", "welcome_msg_classifier_prompt", None)
feedback_msg_classifier_prompt = get_config_value("llm", "feedback_msg_classifier_prompt", None)

learner_ai_base_url = get_config_value('learning', 'learner_ai_base_url', None)
generate_virtual_id_api = get_config_value('learning', 'generate_virtual_id_api', None)
get_milestone_api = get_config_value('learning', 'get_milestone_api', None)
get_learner_profile_api = get_config_value('learning', 'get_user_progress_api', None)
add_lesson_api = get_config_value('learning', 'add_lesson_api', None)
update_learner_profile_api = get_config_value('learning', 'update_learner_profile', None)
get_assessment_api = get_config_value('learning', 'get_assessment_api', None)
get_practice_showcase_contents_api = get_config_value('learning', 'get_practice_showcase_contents_api', None)
get_result_api = get_config_value('learning', 'get_result_api', None)

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

language_code_list = get_config_value('learning', 'supported_lang_codes', None).split(",")
if language_code_list is None:
    raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

learning_language_list = get_config_value('learning', 'learn_language', None)
if learning_language_list is None:
    raise HTTPException(status_code=422, detail="learn_language not configured!")

positive_emotions = get_config_value("learning", "positive_emotions", None).split(",")
other_emotions = get_config_value("learning", "other_emotions", None).split(",")

welcome_msg = json.loads(get_config_value("conversation_messages", "welcome_message", None))
greeting_positive_resp_msg = json.loads(get_config_value("conversation_messages", "greeting_positive_response_message", None))
greeting_other_resp_msg = json.loads(get_config_value("conversation_messages", "greeting_other_response_message", None))
non_greeting_positive_resp_msg = json.loads(get_config_value("conversation_messages", "non_greeting_positive_response_message", None))
non_greeting_other_resp_msg = json.loads(get_config_value("conversation_messages", "non_greeting_other_response_message", None))
get_user_feedback_msg = json.loads(get_config_value("conversation_messages", "get_user_feedback_message", None))
feedback_positive_resp_msg = json.loads(get_config_value("conversation_messages", "feedback_positive_response_message", None))
feedback_other_resp_msg = json.loads(get_config_value("conversation_messages", "feedback_other_response_message", None))
non_feedback_positive_resp_msg = json.loads(get_config_value("conversation_messages", "non_feedback_positive_response_message", None))
non_feedback_other_resp_msg = json.loads(get_config_value("conversation_messages", "non_feedback_other_response_message", None))
conclusion_msg = json.loads(get_config_value("conversation_messages", "conclusion_message", None))
discovery_start_msg = json.loads(get_config_value("conversation_messages", "discovery_phase_message", None))
practice_start_msg = json.loads(get_config_value("conversation_messages", "practice_phase_message", None))
showcase_start_msg = json.loads(get_config_value("conversation_messages", "showcase_phase_message", None))
learning_next_content_msg = json.loads(get_config_value("conversation_messages", "learning_next_content_message", None))


# Define a function to store and retrieve data in Redis
def store_data(key, value):
    redis_client.set(key, value)


def retrieve_data(key):
    data_from_redis = redis_client.get(key)
    return data_from_redis.decode('utf-8') if data_from_redis is not None else None


def remove_data(key):
    redis_client.delete(key)


@app.on_event("startup")
async def startup_event():
    logger.info('Invoking startup_event')
    load_dotenv()
    logger.info('startup_event : Engine created')


@app.on_event("shutdown")
async def shutdown_event():
    logger.info('Invoking shutdown_event')
    logger.info('shutdown_event : Engine closed')


class LoginRequest(BaseModel):
    user_id: str = None
    password: str = None
    conversation_language: str = None
    learning_language: str = None


class LoginResponse(BaseModel):
    user_virtual_id: str = None
    session_id: str = None


class ConversationStartRequest(BaseModel):
    user_virtual_id: str = None


class ConversationRequest(BaseModel):
    user_virtual_id: str = None
    user_audio_msg: str = None


class BotResponse(BaseModel):
    audio: str = None
    state: int = None


class ConversationResponse(BaseModel):
    conversation: BotResponse = None


class ContentResponse(BaseModel):
    audio: str = None
    text: str = None
    content_id: str = None
    milestone: str = None
    milestone_level: str = None


class LearningStartRequest(BaseModel):
    user_virtual_id: str = None


class LearningNextRequest(BaseModel):
    user_virtual_id: str = None
    user_audio_msg: str = None
    content_id: str = None
    original_content_text: str = None


class LearningResponse(BaseModel):
    conversation: BotResponse = None
    content: ContentResponse = None


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"


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


def invoke_llm(user_virtual_id: str, user_statement: str, prompt: str, session_id: str, language: str) -> str:
    logger.info({"intent_classifier": "classifier_prompt", "user_virtual_id": user_virtual_id, "language": language, "session_id": session_id, "user_statement": user_statement})
    res = llm_client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_statement}
        ],
    )
    message = res.choices[0].message.model_dump()
    llm_response = message["content"]
    logger.info({"intent_classifier": "openai_response", "user_virtual_id": user_virtual_id, "language": language, "session_id": session_id, "response": llm_response})
    return llm_response


def emotions_classifier(user_virtual_id: str, user_statement: str, session_id: str, language: str) -> str:
    logger.info({"user_virtual_id": user_virtual_id, "language": language, "session_id": session_id, "user_statement": user_statement})
    user_session_emotions = retrieve_data(user_virtual_id + "_" + language + "_" + session_id + "_emotions")
    logger.info({"user_virtual_id": user_virtual_id, "language": language, "session_id": session_id, "user_session_emotions": user_session_emotions})

    emotion_category = invoke_llm(user_virtual_id, user_statement, emotion_classifier_prompt, session_id, language)
    logger.info({"emotions_classifier": user_virtual_id, "user_statement": user_statement, "emotion_classifier_prompt": emotion_classifier_prompt, "emotion_category": emotion_category, "session_id": session_id, "language": language})
    if user_session_emotions:
        user_session_emotions = json.loads(user_session_emotions)
        user_session_emotions.append(emotion_category)
    else:
        user_session_emotions = [emotion_category]

    store_data(user_virtual_id + "_" + language + "_" + session_id + "_emotions", json.dumps(user_session_emotions))
    return emotion_category


def emotions_summary(emotion_category: str) -> str:
    if emotion_category in positive_emotions:
        return "positive"
    else:
        return "other"


def validate_user(user_virtual_id: str):
    if user_virtual_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid user_virtual_id input!")

    user_learning_language = retrieve_data(user_virtual_id + "_learning_language")
    user_conversation_language = retrieve_data(user_virtual_id + "_conversation_language")
    user_session_id = retrieve_data(user_virtual_id + "_" + user_learning_language + "_session")

    if user_session_id is None or user_learning_language is None or user_conversation_language is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="User session not found!")
    logger.info({"user_virtual_id": user_virtual_id, "user_session_id": user_session_id, "user_learning_language": user_learning_language, "user_conversation_language": user_conversation_language})
    return user_session_id, user_learning_language, user_conversation_language


@app.post("/v1/login", include_in_schema=True)
async def user_login(request: LoginRequest) -> LoginResponse:
    user_id = request.user_id
    password = request.password
    conversation_language = request.conversation_language.strip().lower()
    learning_language = request.learning_language.strip().lower()

    if user_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid user_id input!")

    if password is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid password input!")

    if conversation_language is None or conversation_language == "" or conversation_language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported conversation language code entered!")

    if learning_language is None or learning_language == "" or learning_language not in learning_language_list:
        raise HTTPException(status_code=422, detail="Unsupported learning language code entered!")

    user_virtual_id_resp = requests.request("GET", learner_ai_base_url + generate_virtual_id_api, params={"username": user_id, "password": password})
    if user_virtual_id_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User virtual id generation failed!")

    user_virtual_id = str(json.loads(user_virtual_id_resp.text)["virtualID"])
    store_data(user_virtual_id, user_id)
    store_data(user_virtual_id + "_learning_language", learning_language)
    store_data(user_virtual_id + "_conversation_language", conversation_language)

    # Get milestone of the user
    user_milestone_level_resp = requests.request("GET", learner_ai_base_url + get_milestone_api + user_virtual_id, params={"language": learning_language})
    # {status: "success", data: {milestone_level: "m1"}}
    logger.info("user_milestone_level_resp:: ", user_milestone_level_resp.status_code, " || ", user_milestone_level_resp.text)
    print(json.loads(user_milestone_level_resp.text)["status"])
    if user_milestone_level_resp.status_code != 200 and user_milestone_level_resp.status_code != 201:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User milestone level retrieval failed!")

    user_milestone_level = json.loads(user_milestone_level_resp.text)["data"]["milestone_level"]
    store_data(user_virtual_id + "_" + learning_language + "_milestone_level", user_milestone_level)

    # Get Lesson Progress of the user
    user_progress_resp = requests.request("GET", learner_ai_base_url + get_learner_profile_api + user_virtual_id, params={"language": learning_language})
    if user_progress_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User lesson progress retrieval failed!")

    try:
        user_progress = json.loads(user_progress_resp.text)["result"]["result"]
        store_data(user_virtual_id + "_" + learning_language + "_learning_phase", user_progress["milestone"])
        store_data(user_virtual_id + "_" + learning_language + "_session", user_progress["sessionId"])
    except Exception as e:
        logger.error({"user_virtual_id": user_virtual_id, "error": e})
        store_data(user_virtual_id + "_" + learning_language + "_learning_phase", "discovery")

    current_session_id = retrieve_data(user_virtual_id + "_" + learning_language + "_session")
    if current_session_id is None:
        milliseconds = round(time.time() * 1000)
        current_session_id = user_virtual_id + str(milliseconds)
        store_data(user_virtual_id + "_" + learning_language + "_session", current_session_id)
    logger.info({"user_virtual_id": user_virtual_id, "current_session_id": current_session_id})

    return LoginResponse(user_virtual_id=user_virtual_id, session_id=current_session_id)


@app.post("/v1/welcome_start", include_in_schema=True)
async def welcome_conversation_start(request: ConversationStartRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    validate_user(user_virtual_id)
    conversation_language = retrieve_data(user_virtual_id + "_conversation_language")
    return_welcome_msg = welcome_msg[conversation_language]
    return ConversationResponse(conversation=BotResponse(audio=return_welcome_msg, state=0))


@app.post("/v1/welcome_next", include_in_schema=True)
async def welcome_conversation_next(request: ConversationRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)
    audio = request.user_audio_msg
    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")

    user_statement_reg, user_statement, error_message = process_incoming_voice(audio, user_conversation_language)
    logger.info({"user_virtual_id": user_virtual_id, "audio_converted_eng_text:": user_statement})
    # classify welcome_user_resp emotion into ['Excited', 'Happy', 'Curious', 'Bored', 'Confused', 'Angry', 'Sad']
    emotion_category = emotions_classifier(user_virtual_id, user_statement, user_session_id, user_learning_language)
    emotion_summary = emotions_summary(emotion_category)
    # classify welcome_user_resp intent into 'greeting' and 'other'
    user_intent = invoke_llm(user_virtual_id, user_statement, welcome_msg_classifier_prompt, user_session_id, user_learning_language)
    # Based on the intent, return response
    if user_intent == "greeting" and emotion_summary == "positive":
        return_welcome_intent_msg = greeting_positive_resp_msg[user_conversation_language]
    elif user_intent == "other" and emotion_summary == "positive":
        return_welcome_intent_msg = non_greeting_positive_resp_msg[user_conversation_language]
    if user_intent == "greeting" and emotion_summary == "other":
        return_welcome_intent_msg = greeting_other_resp_msg[user_conversation_language]
    else:
        return_welcome_intent_msg = non_greeting_other_resp_msg[user_conversation_language]

    logger.info({"user_virtual_id": user_virtual_id, "x_session_id": user_session_id, "return_welcome_intent_msg": return_welcome_intent_msg})
    return ConversationResponse(conversation=BotResponse(audio=return_welcome_intent_msg, state=0))


@app.post("/v1/learning_start", include_in_schema=True)
async def learning_conversation_start(request: LearningStartRequest) -> LearningResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)

    # Based on the phase get the content from the collection to be displayed
    user_milestone_level = retrieve_data(user_virtual_id + "_" + user_learning_language + "_milestone_level")
    user_learning_phase = retrieve_data(user_virtual_id + "_" + user_learning_language + "_learning_phase")

    if user_learning_phase == "discovery":
        conversation_message = discovery_start_msg[user_conversation_language]
        # content_response = get_discovery_content(user_milestone_level, user_virtual_id, user_learning_language, user_session_id)
    elif user_learning_phase == "practice":
        conversation_message = practice_start_msg[user_conversation_language]
        # content_response = get_showcase_content(user_virtual_id, user_learning_language, user_session_id)
    elif user_learning_phase == "showcase":
        conversation_message = showcase_start_msg[user_conversation_language]
        # content_response = get_showcase_content(user_virtual_id, user_learning_language, user_session_id)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid learning phase!")

    # Return content information and conversation message

    conversation_response = BotResponse(audio=conversation_message, state=0)
    # content_response = ContentResponse(audio="https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240418-112234.mp3", text="Hello", content_id="hello123")
    content_response = ContentResponse(audio="https://all-dev-content-service.s3.ap-south-1.amazonaws.com/Audio/18d14cba-f1a9-4692-862e-292eb3825e39.wav", text="fun", content_id="18d14cba-f1a9-4692-862e-292eb3825e39", milestone="discovery", milestone_level=user_milestone_level)

    return LearningResponse(conversation=conversation_response, content=content_response)


@app.post("/v1/learning_next", include_in_schema=True)
async def learning_conversation_next(request: LearningNextRequest) -> LearningResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)

    user_milestone_level = retrieve_data(user_virtual_id + "_" + user_learning_language + "_milestone_level")
    user_learning_phase = retrieve_data(user_virtual_id + "_" + user_learning_language + "_learning_phase")

    learning_next_content_message = learning_next_content_msg[user_conversation_language]
    conversation_response = BotResponse(audio=learning_next_content_message, state=0)
    content_response = ContentResponse(audio="https://all-dev-content-service.s3.ap-south-1.amazonaws.com/Audio/b5637e7f-b9ec-4efe-9afc-368e346ef5a9.wav", text="box", content_id="b5637e7f-b9ec-4efe-9afc-368e346ef5a9", milestone="discovery", milestone_level=user_milestone_level)

    return LearningResponse(conversation=conversation_response, content=content_response)


@app.post("/v1/feedback_start", include_in_schema=True)
async def feedback_conversation_start(request: ConversationStartRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    validate_user(user_virtual_id)
    conversation_language = retrieve_data(user_virtual_id + "_conversation_language")
    get_user_feedback_message = get_user_feedback_msg[conversation_language]
    return ConversationResponse(conversation=BotResponse(audio=get_user_feedback_message, state=0))


@app.post("/v1/feedback_next", include_in_schema=True)
async def feedback_conversation_next(request: ConversationRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)
    audio = request.user_audio_msg
    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")

    user_statement_reg, user_statement, error_message = process_incoming_voice(audio, user_conversation_language)
    logger.info({"user_virtual_id": user_virtual_id, "audio_converted_eng_text:": user_statement})
    # classify welcome_user_resp emotion into ['Excited', 'Happy', 'Curious', 'Bored', 'Confused', 'Angry', 'Sad']
    emotion_category = emotions_classifier(user_virtual_id, user_statement, user_session_id, user_learning_language)
    emotion_summary = emotions_summary(emotion_category)
    # classify welcome_user_resp intent into 'greeting' and 'other'
    user_intent = invoke_llm(user_virtual_id, user_statement, feedback_msg_classifier_prompt, user_session_id, user_learning_language)
    # Based on the intent, return response
    if user_intent == "feedback" and emotion_summary == "positive":
        return_feedback_intent_msg = feedback_positive_resp_msg[user_conversation_language]
    elif user_intent == "feedback" and emotion_summary == "other":
        return_feedback_intent_msg = feedback_other_resp_msg[user_conversation_language]
    elif user_intent == "other" and emotion_summary == "positive":
        return_feedback_intent_msg = non_feedback_positive_resp_msg[user_conversation_language]
    else:
        return_feedback_intent_msg = non_feedback_other_resp_msg[user_conversation_language]

    logger.info({"user_virtual_id": user_virtual_id, "x_session_id": user_session_id, "return_feedback_intent_msg": return_feedback_intent_msg})
    return ConversationResponse(conversation=BotResponse(audio=return_feedback_intent_msg, state=0))


@app.post("/v1/conclusion", include_in_schema=True)
async def conclude_session(request: ConversationStartRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    validate_user(user_virtual_id)
    conversation_language = retrieve_data(user_virtual_id + "_conversation_language")
    user_learning_language = retrieve_data(user_virtual_id + "_learning_language")

    # clearing user session details
    remove_data(user_virtual_id + "_" + user_learning_language + "_session")
    # TODO : clear user session learning data

    conclusion_message = conclusion_msg[conversation_language]
    return ConversationResponse(conversation=BotResponse(audio=conclusion_message, state=0))


def generate_sub_session_id(length=24):
    # Define the set of characters to choose from
    characters = string.ascii_letters + string.digits

    # Generate a random session ID
    sub_session_id = ''.join(secrets.choice(characters) for _ in range(length))

    return sub_session_id


def get_discovery_content(user_milestone_level, user_virtual_id, user_learning_language, session_id) -> ContentResponse:
    stored_user_assessment_collections: str = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_collections")

    user_assessment_collections: dict = {}
    if stored_user_assessment_collections:
        user_assessment_collections = json.loads(stored_user_assessment_collections)

    logger.info({"user_virtual_id": user_virtual_id, "Redis user_assessment_collections": user_assessment_collections})

    if stored_user_assessment_collections is None:
        headers = {
            'Content-Type': 'application/json'
        }
        user_assessment_collections: dict = {}
        payload = {"tags": ["ASER"], "language": user_learning_language}

        get_assessment_response = requests.request("POST", get_assessment_api, headers=headers, data=json.dumps(payload))
        logger.info({"user_virtual_id": user_virtual_id, "get_assessment_response": get_assessment_response})

        assessment_data = get_assessment_response.json()["data"]
        logger.info({"user_virtual_id": user_virtual_id, "assessment_data": assessment_data})
        for collection in assessment_data:
            if collection["category"] == "Sentence" or collection["category"] == "Word":
                if user_assessment_collections is None:
                    user_assessment_collections = {collection["category"]: collection}
                elif collection["category"] not in user_assessment_collections.keys():
                    user_assessment_collections.update({collection["category"]: collection})
                elif collection["category"] in user_assessment_collections.keys() and user_milestone_level in collection["tags"]:
                    user_assessment_collections.update({collection["category"]: collection})

        logger.info({"user_virtual_id": user_virtual_id, "user_assessment_collections": json.dumps(user_assessment_collections)})
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_collections", json.dumps(user_assessment_collections))

    completed_collections = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_completed_collections")
    logger.info({"user_virtual_id": user_virtual_id, "completed_collections": completed_collections})
    in_progress_collection = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_progress_collection")
    logger.info({"user_virtual_id": user_virtual_id, "in_progress_collection": in_progress_collection})

    if completed_collections and in_progress_collection and in_progress_collection in json.loads(completed_collections):
        in_progress_collection = None

    if completed_collections:
        completed_collections = json.loads(completed_collections)
        for completed_collection in completed_collections:
            user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != completed_collection}

    current_collection = None

    if in_progress_collection:
        for collection_value in user_assessment_collections.values():
            if collection_value.get("collectionId") == in_progress_collection:
                logger.debug({"user_virtual_id": user_virtual_id, "setting_current_collection_using_in_progress_collection": collection_value})
                current_collection = collection_value
    elif len(user_assessment_collections.values()) > 0:
        current_collection = list(user_assessment_collections.values())[0]
        logger.debug({"user_virtual_id": user_virtual_id, "setting_current_collection_using_assessment_collections": current_collection})
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_progress_collection", current_collection.get("collectionId"))
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_progress_collection_category", current_collection.get("category"))
    else:
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_collections")
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_completed_collections")
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_progress_collection")
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category")
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_completed_contents")
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_session")
        # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_sub_session")
        store_data(user_virtual_id + "_" + user_learning_language + "_" + session_id + "_completed", "true")
        output = ContentResponse(audio="completed", text="completed")
        return output

    logger.info({"user_virtual_id": user_virtual_id, "current_collection": current_collection})

    completed_contents = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_completed_contents")
    logger.debug({"user_virtual_id": user_virtual_id, "completed_contents": completed_contents})
    if completed_contents:
        completed_contents = json.loads(completed_contents)
        for content_id in completed_contents:
            for content in current_collection.get("content"):
                if content.get("contentId") == content_id:
                    current_collection.get("content").remove(content)

    logger.info({"user_virtual_id": user_virtual_id, "updated_current_collection": current_collection})

    if "content" not in current_collection.keys() or len(current_collection.get("content")) == 0:
        if completed_collections:
            completed_collections.append(current_collection.get("collectionId"))
        else:
            completed_collections = [current_collection.get("collectionId")]
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_completed_collections", json.dumps(completed_collections))
        user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != current_collection.get("collectionId")}

        logger.info({"user_virtual_id": user_virtual_id, "completed_collection_id": current_collection.get("collectionId"), "after_removing_completed_collection_user_assessment_collections": user_assessment_collections})


        add_lesson_payload = {"userId": user_virtual_id, "sessionId": session_id, "milestone": "discovery", "lesson": current_collection.get("name"), "progress": 100,
                              "milestoneLevel": user_milestone_level, "language": user_learning_language}
        add_lesson_response = requests.request("POST", learner_ai_base_url + add_lesson_api, headers=headers, data=json.dumps(add_lesson_payload))
        logger.info({"user_virtual_id": user_virtual_id, "add_lesson_response": add_lesson_response})

        if len(user_assessment_collections) != 0:
            current_collection = list(user_assessment_collections.values())[0]
            logger.info({"user_virtual_id": user_virtual_id, "current_collection": current_collection})
            store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_progress_collection", current_collection.get("collectionId"))
        else:
            # get_result_api = get_config_value('learning', 'get_result_api', None)
            # get_result_payload = {"sub_session_id": sub_session_id, "contentType": current_collection.get("category"), "session_id": session_id, "user_virtual_id": user_virtual_id, "collectionId": current_collection.get("collectionId"), "language": language}
            # get_result_response = requests.request("POST", get_result_api, headers=headers, data=json.dumps(get_result_payload))
            # logger.info({"user_virtual_id": user_virtual_id, "get_result_response": get_result_response})
            # percentage = get_result_response.json()["data"]["percentage"]
            #
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_collections")
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_completed_collections")
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_progress_collection")
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category")
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_completed_contents")
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_session")
            # redis_client.delete(user_virtual_id + "_" + language + "_" + user_milestone_level + "_sub_session")
            store_data(user_virtual_id + "_" + user_learning_language + "_" + session_id + "_completed", "true")
            output = ContentResponse(audio="completed", text="completed")
            return output

    content_source_data = current_collection.get("content")[0].get("contentSourceData")[0]
    logger.debug({"user_virtual_id": user_virtual_id, "content_source_data": content_source_data})
    content_id = current_collection.get("content")[0].get("contentId")

    output = ContentResponse(audio=content_source_data.get("audioUrl"), text=content_source_data.get("text"), content_id=content_id)
    return output


def get_showcase_content(user_virtual_id, language, current_session_id) -> ContentResponse:
    current_content = None
    stored_user_showcase_contents: str = retrieve_data(user_virtual_id + "_" + language + "_showcase_contents")
    user_showcase_contents = []
    if stored_user_showcase_contents:
        user_showcase_contents = json.loads(stored_user_showcase_contents)

    logger.info({"user_virtual_id": user_virtual_id, "Redis stored_user_showcase_contents": stored_user_showcase_contents})

    learning_language = get_config_value('request', 'learn_language', None)

    if stored_user_showcase_contents is None:
        get_showcase_contents_api = get_config_value('learning', 'get_showcase_contents_api', None) + user_virtual_id
        content_limit = int(get_config_value('request', 'content_limit', None))
        target_limit = int(get_config_value('request', 'target_limit', None))
        # defining a params dict for the parameters to be sent to the API
        params = {'language': learning_language, 'contentlimit': content_limit, 'gettargetlimit': target_limit}
        # sending get request and saving the response as response object
        showcase_contents_response = requests.get(url=get_showcase_contents_api + user_virtual_id, params=params)
        user_showcase_contents = showcase_contents_response.json()["content"]
        logger.info({"user_virtual_id": user_virtual_id, "user_showcase_contents": user_showcase_contents})
        store_data(user_virtual_id + "_" + language + "_showcase_contents", json.dumps(user_showcase_contents))

    completed_contents = retrieve_data(user_virtual_id + "_" + language + "_completed_contents")
    logger.info({"user_virtual_id": user_virtual_id, "completed_contents": completed_contents})
    in_progress_content = retrieve_data(user_virtual_id + "_" + language + "_progress_content")
    logger.info({"user_virtual_id": user_virtual_id, "progress_content": in_progress_content})

    if completed_contents and in_progress_content and in_progress_content in json.loads(completed_contents):
        in_progress_content = None

    if completed_contents:
        completed_contents = json.loads(completed_contents)
        for completed_content in completed_contents:
            for showcase_content in user_showcase_contents:
                if showcase_content.get("contentId") == completed_content:
                    user_showcase_contents.remove(showcase_content)

    if in_progress_content is None and len(user_showcase_contents) > 0:
        current_content = user_showcase_contents[0]
        store_data(user_virtual_id + "_" + language + "_progress_content", current_content.get("contentId"))
    elif in_progress_content is not None and len(user_showcase_contents) > 0:
        for showcase_content in user_showcase_contents:
            if showcase_content.get("contentId") == in_progress_content:
                current_content = showcase_content
    else:
        # redis_client.delete(user_virtual_id + "_" + language + "_contents")
        # redis_client.delete(user_virtual_id + "_" + language + "_progress_content")
        # redis_client.delete(user_virtual_id + "_" + language + "_completed_contents")
        # redis_client.delete(user_virtual_id + "_" + language + "_session")
        # redis_client.delete(user_virtual_id + "_" + language + "_sub_session")
        store_data(user_virtual_id + "_" + language + "_" + current_session_id + "_completed", "true")
        output = ContentResponse(audio="completed", text="completed")
        return output

    logger.info({"user_virtual_id": user_virtual_id, "current_content": current_content})
    content_source_data = current_content.get("contentSourceData")[0]
    logger.debug({"user_virtual_id": user_virtual_id, "content_source_data": content_source_data})
    content_id = current_content.get("contentId")
    audio_url = "https://all-dev-content-service.s3.ap-south-1.amazonaws.com/Audio/" + content_id + ".wav"

    output = ContentResponse(audio=audio_url, text=content_source_data.get("text"), content_id=content_id)
    return output
