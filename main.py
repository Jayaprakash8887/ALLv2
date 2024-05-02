import secrets
import string

import openai
import redis
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.pydantic_v1 import BaseModel
from io import BytesIO
from io_processing import *
from logger import logger
from utils import is_url, is_base64

gpt_model = get_config_value("llm", "gpt_model", None)

welcome_emotion_classifier_prompt = get_config_value("llm", "welcome_emotion_classifier_prompt", None)
welcome_msg_classifier_prompt = get_config_value("llm", "welcome_msg_classifier_prompt", None)
feedback_emotion_classifier_prompt = get_config_value("llm", "feedback_emotion_classifier_prompt", None)
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
content_limit = int(get_config_value('learning', 'content_limit', None))
target_limit = int(get_config_value('learning', 'target_limit', None))

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
system_not_available_msg = json.loads(get_config_value("conversation_messages", "system_not_available_message", None))

headers = {'Content-Type': 'application/json'}


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


class BotStartResponse(BaseModel):
    audio: str = None


class ConversationStartResponse(BaseModel):
    conversation: BotStartResponse = None


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


class LearningStartResponse(BaseModel):
    conversation: BotStartResponse = None
    content: ContentResponse = None


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


def emotions_classifier(emotion_type: str, user_virtual_id: str, user_statement: str, session_id: str, language: str) -> str:
    logger.info({"user_virtual_id": user_virtual_id, "language": language, "session_id": session_id, "user_statement": user_statement})
    user_session_emotions = retrieve_data(user_virtual_id + "_" + language + "_" + session_id + "_emotions")
    logger.info({"user_virtual_id": user_virtual_id, "language": language, "session_id": session_id, "user_session_emotions": user_session_emotions})

    if emotion_type == "welcome":
        emotion_classifier_prompt = welcome_emotion_classifier_prompt
    else:
        emotion_classifier_prompt = feedback_emotion_classifier_prompt

    emotion_category = invoke_llm(user_virtual_id, user_statement, welcome_emotion_classifier_prompt, session_id, language)
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
    logger.debug({"request": request})
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

    logger.info({"user_id": user_id, "password": password, "conversation_language": conversation_language, "learning_language": learning_language})

    user_virtual_id_resp = requests.request("GET", learner_ai_base_url + generate_virtual_id_api, params={"username": user_id, "password": password})
    if user_virtual_id_resp.status_code != 200 and user_virtual_id_resp.status_code != 201:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User virtual id generation failed!")

    user_virtual_id = str(json.loads(user_virtual_id_resp.text)["virtualID"])
    store_data(user_virtual_id, user_id)
    store_data(user_virtual_id + "_learning_language", learning_language)
    store_data(user_virtual_id + "_conversation_language", conversation_language)

    logger.info({"user_virtual_id": user_virtual_id, "invoking_api": get_milestone_api})

    # Get milestone of the user
    user_milestone_level_resp = requests.request("GET", learner_ai_base_url + get_milestone_api + user_virtual_id, params={"language": learning_language})
    # {status: "success", data: {milestone_level: "m1"}}
    logger.info({"user_virtual_id": user_virtual_id, "user_milestone_level_resp": user_milestone_level_resp.status_code, "text": user_milestone_level_resp.text})
    print(json.loads(user_milestone_level_resp.text)["status"])
    if user_milestone_level_resp.status_code != 200 and user_milestone_level_resp.status_code != 201:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User milestone level retrieval failed!")

    user_milestone_level = json.loads(user_milestone_level_resp.text)["data"]["milestone_level"]
    store_data(user_virtual_id + "_" + learning_language + "_milestone_level", user_milestone_level)

    # Get Lesson Progress of the user
    user_progress_resp = requests.request("GET", learner_ai_base_url + get_learner_profile_api + user_virtual_id, params={"language": learning_language})
    if user_progress_resp.status_code != 200 and user_progress_resp.status_code != 201:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User lesson progress retrieval failed!")

    try:
        user_progress = json.loads(user_progress_resp.text)["result"]["result"]
        current_learning_phase = retrieve_data(user_virtual_id + "_" + learning_language + "_learning_phase")
        if current_learning_phase is None:
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
async def welcome_conversation_start(request: ConversationStartRequest) -> ConversationStartResponse:
    user_virtual_id = request.user_virtual_id
    validate_user(user_virtual_id)
    conversation_language = retrieve_data(user_virtual_id + "_conversation_language")
    return_welcome_msg = welcome_msg[conversation_language]
    return ConversationStartResponse(conversation=BotStartResponse(audio=return_welcome_msg))


@app.post("/v1/welcome_next", include_in_schema=True)
async def welcome_conversation_next(request: ConversationRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)
    audio = request.user_audio_msg
    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")
    state = 0
    try:
        user_statement_reg, user_statement, error_message = process_incoming_voice(audio, user_conversation_language)
        logger.info({"user_virtual_id": user_virtual_id, "audio_converted_eng_text:": user_statement})
        # classify welcome_user_resp emotion into ['Excited', 'Happy', 'Curious', 'Bored', 'Confused', 'Angry', 'Sad']
        emotion_category = emotions_classifier("welcome", user_virtual_id, user_statement, user_session_id, user_learning_language)
        emotion_summary = emotions_summary(emotion_category)
        # classify welcome_user_resp intent into 'greeting' and 'other'
        user_intent = invoke_llm(user_virtual_id, user_statement, welcome_msg_classifier_prompt, user_session_id, user_learning_language)
        if user_intent:
            user_intent = user_intent.lower().replace("\'", "")
        logger.info(
            {"user_virtual_id": user_virtual_id, "user_session_id": user_session_id, "user_statement": user_statement, "welcome_msg_classifier_prompt": welcome_msg_classifier_prompt, "user_intent": user_intent, "emotion_summary": emotion_summary})
        # Based on the intent, return response
        if user_intent == "greeting" and emotion_summary == "positive":
            return_welcome_intent_msg = greeting_positive_resp_msg[user_conversation_language]
        elif user_intent == "other" and emotion_summary == "positive":
            return_welcome_intent_msg = non_greeting_positive_resp_msg[user_conversation_language]
        elif user_intent == "greeting" and emotion_summary == "other":
            return_welcome_intent_msg = greeting_other_resp_msg[user_conversation_language]
        else:
            return_welcome_intent_msg = non_greeting_other_resp_msg[user_conversation_language]
    except Exception as e:
        logger.error(f"Exception while translating audio or invoking llm: {e}", exc_info=True)
        return_welcome_intent_msg = system_not_available_msg[user_conversation_language]
        state = -1

    logger.info({"user_virtual_id": user_virtual_id, "x_session_id": user_session_id, "return_welcome_intent_msg": return_welcome_intent_msg})
    return ConversationResponse(conversation=BotResponse(audio=return_welcome_intent_msg, state=state))


@app.post("/v1/learning_start", include_in_schema=True)
async def learning_conversation_start(request: LearningStartRequest) -> LearningStartResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)

    # Based on the phase get the content from the collection to be displayed
    user_milestone_level = retrieve_data(user_virtual_id + "_" + user_learning_language + "_milestone_level")
    user_learning_phase = retrieve_data(user_virtual_id + "_" + user_learning_language + "_learning_phase")

    phase_session_id = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_learning_phase + "_sub_session")
    if phase_session_id is None:
        phase_session_id = generate_sub_session_id()
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_learning_phase + "_sub_session", phase_session_id)

    if user_learning_phase == "discovery":
        conversation_message = discovery_start_msg[user_conversation_language]
    elif user_learning_phase == "practice":
        conversation_message = practice_start_msg[user_conversation_language]
    elif user_learning_phase == "showcase":
        conversation_message = showcase_start_msg[user_conversation_language]
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid learning phase!")

    logger.info({"user_virtual_id": user_virtual_id, "user_session_id": user_session_id, "user_milestone_level": user_milestone_level, "user_learning_phase": user_learning_phase, "phase_session_id": phase_session_id,
                 "conversation_message": conversation_message})

    # Return content information and conversation message
    content_response = fetch_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
    conversation_response = BotStartResponse(audio=conversation_message)
    logger.info({"user_virtual_id": user_virtual_id, "conversation_response": conversation_response, "content_response": content_response})
    return LearningStartResponse(conversation=conversation_response, content=content_response)


@app.post("/v1/learning_next", include_in_schema=True)
async def learning_conversation_next(request: LearningNextRequest) -> LearningResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)

    user_audio = request.user_audio_msg
    if not is_url(user_audio) and not is_base64(user_audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid user audio input!")
    logger.debug({"user_virtual_id": user_virtual_id, "user_session_id": user_session_id, "user_audio": user_audio})

    content_id = request.content_id
    if not content_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid content_id input!")

    original_content_text = request.original_content_text
    if not original_content_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid original_content_text input!")

    user_milestone_level = retrieve_data(user_virtual_id + "_" + user_learning_language + "_milestone_level")
    user_learning_phase = retrieve_data(user_virtual_id + "_" + user_learning_language + "_learning_phase")
    phase_session_id = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_learning_phase + "_sub_session")
    in_progress_collection_category = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection_category")

    logger.info({"user_virtual_id": user_virtual_id, "user_session_id": user_session_id, "user_milestone_level": user_milestone_level, "user_learning_phase": user_learning_phase, "phase_session_id": phase_session_id,
                 "in_progress_collection_category": in_progress_collection_category})

    # Submit user response if the phase is not 'practice'
    if user_learning_phase != "practice":
        # Get the current date and Format the date as "YYYY-MM-DD"
        current_date = datetime.now().date()
        formatted_date = current_date.strftime("%Y-%m-%d")

        if is_url(user_audio):
            local_filename = generate_temp_filename("mp3")
            with requests.get(user_audio) as r:
                with open(local_filename, 'wb') as f:
                    f.write(r.content)
            output_file = AudioSegment.from_file(local_filename)
            mp3_output_file = output_file.export(local_filename, format="mp3")
            given_audio = AudioSegment.from_file(mp3_output_file)
            given_audio_bytes = given_audio.export().read()
            user_audio = base64.b64encode(given_audio_bytes).decode('utf-8')
            os.remove(local_filename)
        else:
            # Decode base64 audio string
            audio_bytes = base64.b64decode(user_audio)
            # Load audio from bytes
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            # Convert audio to MP3 format
            output_buffer = BytesIO()
            audio.export(output_buffer, format="mp3")
            # Get base64 string of MP3 audio
            user_audio = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        payload = {"audio": user_audio, "contentId": content_id, "contentType": in_progress_collection_category, "date": formatted_date, "language": user_learning_language, "original_text": original_content_text, "session_id": user_session_id,
                   "sub_session_id": phase_session_id, "user_id": user_virtual_id}

        logger.info({"user_virtual_id": user_virtual_id, "update_learner_profile_payload": payload})
        update_learner_profile_response = requests.request("POST", learner_ai_base_url + update_learner_profile_api + user_learning_language, headers=headers, data=json.dumps(payload))
        logger.info({"user_virtual_id": user_virtual_id, "update_learner_profile_response": update_learner_profile_response.status_code})
        if update_learner_profile_response.status_code != 200 and update_learner_profile_response.status_code != 201:
            raise HTTPException(500, "Submitted response could not be registered!")
        update_status = update_learner_profile_response.json()["status"]

        if update_status == "success":
            completed_contents = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents")
            if completed_contents:
                completed_contents = json.loads(completed_contents)
                if type(completed_contents) == list:
                    completed_contents = set(completed_contents)
                completed_contents.add(content_id)
            else:
                completed_contents = {content_id}
            completed_contents = list(completed_contents)
            logger.debug({"user_virtual_id": user_virtual_id, "updated_completed_contents": completed_contents})
            store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents", json.dumps(completed_contents))
        else:
            raise HTTPException(500, "Submitted response could not be registered!")
    elif user_learning_phase == "practice":
        completed_contents = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents")
        if completed_contents:
            completed_contents = json.loads(completed_contents)
            if type(completed_contents) == list:
                completed_contents = set(completed_contents)
            completed_contents.add(content_id)
        else:
            completed_contents = {content_id}
        completed_contents = list(completed_contents)
        logger.debug({"user_virtual_id": user_virtual_id, "updated_completed_contents": completed_contents})
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents", json.dumps(completed_contents))

    learning_next_content_message = learning_next_content_msg[user_conversation_language]
    content_response = fetch_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)

    if content_response is not None and content_response.text:
        conversation_response = BotResponse(audio=learning_next_content_message, state=1)
    else:
        conversation_response = BotResponse(state=0)
    logger.info({"user_virtual_id": user_virtual_id, "conversation_response": conversation_response, "content_response": content_response})
    return LearningResponse(conversation=conversation_response, content=content_response)


@app.post("/v1/feedback_start", include_in_schema=True)
async def feedback_conversation_start(request: ConversationStartRequest) -> ConversationStartResponse:
    user_virtual_id = request.user_virtual_id
    validate_user(user_virtual_id)
    conversation_language = retrieve_data(user_virtual_id + "_conversation_language")
    get_user_feedback_message = get_user_feedback_msg[conversation_language]
    return ConversationStartResponse(conversation=BotStartResponse(audio=get_user_feedback_message))


@app.post("/v1/feedback_next", include_in_schema=True)
async def feedback_conversation_next(request: ConversationRequest) -> ConversationResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)
    audio = request.user_audio_msg
    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")

    state = 0
    try:
        user_statement_reg, user_statement, error_message = process_incoming_voice(audio, user_conversation_language)
        logger.info({"user_virtual_id": user_virtual_id, "audio_converted_eng_text:": user_statement})
        # classify welcome_user_resp emotion into ['Excited', 'Happy', 'Curious', 'Bored', 'Confused', 'Angry', 'Sad']
        emotion_category = emotions_classifier("feedbck", user_virtual_id, user_statement, user_session_id, user_learning_language)
        emotion_summary = emotions_summary(emotion_category)
        # classify welcome_user_resp intent into 'greeting' and 'other'
        user_intent = invoke_llm(user_virtual_id, user_statement, feedback_msg_classifier_prompt, user_session_id, user_learning_language)
        if user_intent:
            user_intent = user_intent.lower().replace("\'", "")
        logger.info(
            {"user_virtual_id": user_virtual_id, "user_session_id": user_session_id, "user_statement": user_statement, "feedback_msg_classifier_prompt": feedback_msg_classifier_prompt, "user_intent": user_intent, "emotion_summary": emotion_summary})
        # Based on the intent, return response
        if user_intent == "feedback" and emotion_summary == "positive":
            return_feedback_intent_msg = feedback_positive_resp_msg[user_conversation_language]
        elif user_intent == "feedback" and emotion_summary == "other":
            return_feedback_intent_msg = feedback_other_resp_msg[user_conversation_language]
        elif user_intent == "other" and emotion_summary == "positive":
            return_feedback_intent_msg = non_feedback_positive_resp_msg[user_conversation_language]
        else:
            return_feedback_intent_msg = non_feedback_other_resp_msg[user_conversation_language]
    except Exception as e:
        logger.error(f"Exception while translating audio or invoking llm: {e}", exc_info=True)
        return_feedback_intent_msg = system_not_available_msg[user_conversation_language]
        state = -1

    logger.info({"user_virtual_id": user_virtual_id, "x_session_id": user_session_id, "return_feedback_intent_msg": return_feedback_intent_msg})
    return ConversationResponse(conversation=BotResponse(audio=return_feedback_intent_msg, state=state))


@app.post("/v1/conclusion", include_in_schema=True)
async def conclude_session(request: ConversationStartRequest) -> ConversationStartResponse:
    user_virtual_id = request.user_virtual_id
    user_session_id, user_learning_language, user_conversation_language = validate_user(user_virtual_id)

    # clearing user session details
    remove_data(user_virtual_id + "_" + user_learning_language + "_session")

    conclusion_message = conclusion_msg[user_conversation_language]
    return ConversationStartResponse(conversation=BotStartResponse(audio=conclusion_message))


def generate_sub_session_id(length=24):
    # Define the set of characters to choose from
    characters = string.ascii_letters + string.digits

    # Generate a random session ID
    sub_session_id = ''.join(secrets.choice(characters) for _ in range(length))

    return sub_session_id


def fetch_content(user_virtual_id: str, user_milestone_level: str, user_learning_phase: str, user_learning_language: str, user_session_id: str, phase_session_id: str) -> ContentResponse:
    if user_learning_phase == "discovery":
        content_response = get_assessment(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
    elif user_learning_phase == "practice":
        content_response = get_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
    elif user_learning_phase == "showcase":
        content_response = get_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid learning phase!")

    return content_response


def shift_to_next_phase(user_virtual_id: str, user_milestone_level: str, user_learning_phase: str, user_learning_language: str, user_session_id: str, phase_session_id: str, in_progress_collection: str,
                        in_progress_collection_category: str) -> ContentResponse:
    logger.debug({"user_virtual_id": user_virtual_id, "user_milestone_level": user_milestone_level, "user_learning_phase": user_learning_phase, "user_learning_language": user_learning_language, "user_session_id": user_session_id,
                  "phase_session_id": phase_session_id, "in_progress_collection": in_progress_collection, "in_progress_collection_category": in_progress_collection_category})
    if user_learning_phase == "discovery":
        get_set_result_resp = requests.request("POST", learner_ai_base_url + get_result_api, headers=headers, data=json.dumps(
            {"sub_session_id": phase_session_id, "contentType": in_progress_collection_category, "session_id": user_session_id, "user_id": user_virtual_id, "collectionId": in_progress_collection, "language": user_learning_language}))
        logger.info({"user_virtual_id": user_virtual_id, "get_set_result_resp": get_set_result_resp})

        if get_set_result_resp.status_code != 200 and get_set_result_resp and get_set_result_resp.json()["status"] != "success":
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Get result API failed!")

        user_milestone_level = get_set_result_resp.json()["data"]["currentLevel"]
        store_data(user_virtual_id + "_" + user_learning_language + "_milestone_level", user_milestone_level)
        session_result = get_set_result_resp.json()["data"]["sessionResult"]
        logger.debug({"user_virtual_id": user_virtual_id, "user_milestone_level": user_milestone_level, "session_result": session_result})
        if session_result == "pass":
            user_learning_phase = "practice"
            store_data(user_virtual_id + "_" + user_learning_language + "_learning_phase", user_learning_phase)
            # return get_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
        # TODO - write logic for fail scenario. Do we restart discovery with Character collection set?
        elif session_result == "fail":
            user_learning_phase = "practice"
            store_data(user_virtual_id + "_" + user_learning_language + "_learning_phase", user_learning_phase)
            # return get_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
        logger.debug({"user_virtual_id": user_virtual_id, "updated_user_milestone_level": user_milestone_level, "updated_user_learning_phase": user_learning_phase})
    elif user_learning_phase == "practice":
        user_learning_phase = "showcase"
        store_data(user_virtual_id + "_" + user_learning_language + "_learning_phase", user_learning_phase)
        # return get_content(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id)
    elif user_learning_phase == "showcase":
        get_set_result_resp = requests.request("POST", learner_ai_base_url + get_result_api, headers=headers, data=json.dumps(
            {"sub_session_id": phase_session_id, "contentType": in_progress_collection_category, "session_id": user_session_id, "user_id": user_virtual_id, "collectionId": in_progress_collection, "language": user_learning_language}))
        logger.info({"user_virtual_id": user_virtual_id, "get_set_result_resp": get_set_result_resp})

        if get_set_result_resp.status_code != 200 and get_set_result_resp and get_set_result_resp.json()["status"] != "success":
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Get result API failed!")

        user_milestone_level = get_set_result_resp.json()["data"]["currentLevel"]
        store_data(user_virtual_id + "_" + user_learning_language + "_milestone_level", user_milestone_level)
        store_data(user_virtual_id + "_" + user_learning_language + "_learning_phase", "discovery")
        session_result = get_set_result_resp.json()["data"]["sessionResult"]

        # TODO - write logic for pass and fail scenario. Do we restart discovery?

        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection")
        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection_category")
        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents")
        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_collections")
        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_sub_session")
        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_collections")
        remove_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_content")
    else:
        # return ContentResponse()
        logger.info("shifting to next phase")

    return ContentResponse()


def get_assessment(user_virtual_id: str, user_milestone_level: str, user_learning_phase: str, user_learning_language: str, user_session_id: str, phase_session_id: str) -> ContentResponse:
    stored_user_assessment_collections: str = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_collections")

    user_assessment_collections: dict = {}
    if stored_user_assessment_collections:
        user_assessment_collections = json.loads(stored_user_assessment_collections)

    logger.info({"user_virtual_id": user_virtual_id, "user_assessment_collections": user_assessment_collections})

    if stored_user_assessment_collections is None:
        user_assessment_collections: dict = {}
        payload = {"tags": ["ASER"], "language": user_learning_language}

        get_assessment_response = requests.request("POST", learner_ai_base_url + get_assessment_api, headers=headers, data=json.dumps(payload))
        logger.info({"user_virtual_id": user_virtual_id, "get_assessment_response": get_assessment_response})
        if get_assessment_response.status_code != 200 and get_assessment_response.status_code != 201:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Assessment collections retrieval failed!")
        assessment_data = get_assessment_response.json()["data"]
        logger.info({"user_virtual_id": user_virtual_id, "assessment_data": assessment_data})
        for collection in assessment_data:
            if collection["category"] == "Word":
                if user_assessment_collections is None:
                    user_assessment_collections = {collection["category"]: collection}
                elif collection["category"] not in user_assessment_collections.keys():
                    user_assessment_collections.update({collection["category"]: collection})
                elif collection["category"] in user_assessment_collections.keys() and user_milestone_level in collection["tags"]:
                    user_assessment_collections.update({collection["category"]: collection})

        logger.info({"user_virtual_id": user_virtual_id, "user_assessment_collections": json.dumps(user_assessment_collections)})
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_collections", json.dumps(user_assessment_collections))

    completed_collections = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_collections")
    logger.info({"user_virtual_id": user_virtual_id, "completed_collections": completed_collections})
    in_progress_collection = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection")
    logger.info({"user_virtual_id": user_virtual_id, "in_progress_collection": in_progress_collection})

    in_progress_collection_category = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection_category")
    logger.info({"user_virtual_id": user_virtual_id, "in_progress_collection_category": in_progress_collection_category})

    if completed_collections and in_progress_collection and in_progress_collection in json.loads(completed_collections):
        return shift_to_next_phase(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id, in_progress_collection, in_progress_collection_category)
        # in_progress_collection = None

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
        in_progress_collection = current_collection.get("collectionId")
        in_progress_collection_category = current_collection.get("category")
        logger.debug({"user_virtual_id": user_virtual_id, "setting_current_collection_using_assessment_collections": current_collection})
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection", in_progress_collection)
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection_category", in_progress_collection_category)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No assessment collections found!")

    logger.info({"user_virtual_id": user_virtual_id, "current_collection": current_collection})

    completed_contents = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents")
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
            completed_collections = json.loads(completed_collections)
            completed_collections.append(current_collection.get("collectionId"))
        else:
            completed_collections = [current_collection.get("collectionId")]
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_collections", json.dumps(completed_collections))
        user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != current_collection.get("collectionId")}
        logger.info({"user_virtual_id": user_virtual_id, "completed_collection_id": current_collection.get("collectionId"), "after_removing_completed_collection_user_assessment_collections": user_assessment_collections})

        if len(user_assessment_collections) != 0:
            current_collection = list(user_assessment_collections.values())[0]
            logger.info({"user_virtual_id": user_virtual_id, "current_collection": current_collection})
            store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection", current_collection.get("collectionId"))
        else:
            return shift_to_next_phase(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id, in_progress_collection, in_progress_collection_category)

    content_source_data = current_collection.get("content")[0].get("contentSourceData")[0]
    logger.debug({"user_virtual_id": user_virtual_id, "content_source_data": content_source_data})
    content_id = current_collection.get("content")[0].get("contentId")
    if completed_contents:
        add_lesson_payload = {"userId": user_virtual_id, "sessionId": user_session_id, "milestone": "discovery", "lesson": current_collection.get("name"),
                              "progress": (len(completed_contents) + 1) / (len(current_collection.get("content")) + len(completed_contents)) * 100,
                              "milestoneLevel": user_milestone_level, "language": user_learning_language}
    else:
        add_lesson_payload = {"userId": user_virtual_id, "sessionId": user_session_id, "milestone": "discovery", "lesson": current_collection.get("name"),
                              "progress": 1 / (len(current_collection.get("content"))) * 100,
                              "collectionId": current_collection.get("collectionId"), "milestoneLevel": user_milestone_level, "language": user_learning_language}
    logger.info({"user_virtual_id": user_virtual_id, "add_lesson_payload": add_lesson_payload})
    add_lesson_response = requests.request("POST", learner_ai_base_url + add_lesson_api, headers=headers, data=json.dumps(add_lesson_payload))
    logger.info({"user_virtual_id": user_virtual_id, "add_lesson_response": add_lesson_response})

    output = ContentResponse(audio=content_source_data.get("audioUrl"), text=content_source_data.get("text"), content_id=content_id, milestone_level=user_milestone_level, milestone=user_learning_phase)
    return output


def get_content(user_virtual_id: str, user_milestone_level: str, user_learning_phase: str, user_learning_language: str, user_session_id: str, phase_session_id: str) -> ContentResponse:
    current_content = None
    stored_user_practice_showcase_contents: str = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_contents")
    user_showcase_contents = []
    if stored_user_practice_showcase_contents:
        user_showcase_contents = json.loads(stored_user_practice_showcase_contents)

    logger.info({"user_virtual_id": user_virtual_id, "Redis stored_user_showcase_contents": stored_user_practice_showcase_contents})

    if stored_user_practice_showcase_contents is None:
        # defining a params dict for the parameters to be sent to the API
        params = {'language': user_learning_language, 'contentlimit': content_limit, 'gettargetlimit': target_limit}
        # sending get request and saving the response as response object
        get_contents_response = requests.request("GET", learner_ai_base_url + get_practice_showcase_contents_api + "word/" + user_virtual_id, params=params)
        logger.info({"user_virtual_id": user_virtual_id, "get_contents_response": get_contents_response})
        if get_contents_response.status_code != 200 and get_contents_response.status_code != 201:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Learner Contents retrieval failed!")

        user_showcase_contents = get_contents_response.json()["content"]
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_contents", json.dumps(user_showcase_contents))

    completed_contents = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_completed_contents")
    logger.info({"user_virtual_id": user_virtual_id, "completed_contents": completed_contents})
    in_progress_content = retrieve_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_content")
    logger.info({"user_virtual_id": user_virtual_id, "progress_content": in_progress_content})

    if completed_contents and in_progress_content and in_progress_content in json.loads(completed_contents):
        in_progress_content = None

    if completed_contents:
        completed_contents = json.loads(completed_contents)
        for completed_content in completed_contents:
            for showcase_content in user_showcase_contents:
                if showcase_content.get("contentId") == completed_content:
                    user_showcase_contents.remove(showcase_content)

    logger.debug({"user_virtual_id": user_virtual_id, "user_showcase_contents": user_showcase_contents, "in_progress_content": in_progress_content})

    if in_progress_content is None and len(user_showcase_contents) > 0:
        current_content = user_showcase_contents[0]
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_content", current_content.get("contentId"))
        store_data(user_virtual_id + "_" + user_learning_language + "_" + user_milestone_level + "_" + user_learning_phase + "_progress_collection_category", current_content.get("contentType"))
    elif in_progress_content is not None and len(user_showcase_contents) > 0:
        for showcase_content in user_showcase_contents:
            if showcase_content.get("contentId") == in_progress_content:
                current_content = showcase_content
    else:
        return shift_to_next_phase(user_virtual_id, user_milestone_level, user_learning_phase, user_learning_language, user_session_id, phase_session_id, "", "")

    logger.info({"user_virtual_id": user_virtual_id, "current_content": current_content})
    content_source_data = current_content.get("contentSourceData")[0]
    logger.debug({"user_virtual_id": user_virtual_id, "content_source_data": content_source_data})
    content_id = current_content.get("contentId")

    # make addLesson call
    if completed_contents:
        add_lesson_payload = {"userId": user_virtual_id, "sessionId": user_session_id, "milestone": user_learning_phase, "lesson": "0",
                              "progress": (len(completed_contents) + 1) / content_limit * 100, "milestoneLevel": user_milestone_level, "language": user_learning_language}
    else:
        add_lesson_payload = {"userId": user_virtual_id, "sessionId": user_session_id, "milestone": "discovery", "lesson": "0",
                              "progress": 1 / content_limit * 100, "milestoneLevel": user_milestone_level, "language": user_learning_language}
    logger.info({"user_virtual_id": user_virtual_id, "add_lesson_payload": add_lesson_payload})
    add_lesson_response = requests.request("POST", learner_ai_base_url + add_lesson_api, headers=headers, data=json.dumps(add_lesson_payload))
    logger.info({"user_virtual_id": user_virtual_id, "add_lesson_response": add_lesson_response})

    if user_learning_phase == "practice":
        audio_url = "https://all-dev-content-service.s3.ap-south-1.amazonaws.com/Audio/" + content_id + ".wav"
    else:
        audio_url = content_source_data.get("audioUrl")

    output = ContentResponse(audio=audio_url, text=content_source_data.get("text"), content_id=content_id, milestone_level=user_milestone_level, milestone=user_learning_phase)
    return output
