-----------------STT----------------------

#Model Download
cd  stt_service
pip install -r model/requirements.txt
export CT2_QUANT_PRIMARY= float16 
export CT2_QUANT_FALLBACK= float16 
python model/model_download_and_convert.py

#Model Running
pip install -r requirements.txt
CONFIG_PATH=stt_primary.env  
uvicorn stt_primary:app  --host 0.0.0.0 --port 8081 --workers 1
CONFIG_PATH=stt_fallback.env  
uvicorn stt_fallback:app  --host 0.0.0.0 --port 8082 --workers 1

python stt_test.py \
  --file /Users/sankalppatidar/Developers/Voice\ Library/Test_Code/test_wav/F10_02_04.wav \
  --base-url http://127.0.0.1:8081 \
  --api-key primary_key_abc \
  --language en

-----------------LLM----------------------

export GROQ_API_KEY=""
export GROQ_API_BASE=https://api.groq.com/openai
python llm_test.py --chat-model llama-3.3-70b-versatile --fast-model llama-3.1-8b-instant

-----------------TTS----------------------

python snac_worker/download_model.py 

uvicorn tts_service.app:app --host 0.0.0.0 --port 8080
uvicorn snac_worker.app:app --host 0.0.0.0 --port 8085 --workers 1

python tts_test.py \
  --urls http://localhost:8080 \
  --api-key dev_key \
  --text "Streaming test. Low latency please." \
  --voice default \
  --out out.wav \
  --verbose

python tts_test.py \
--urls http://localhost:8080 \
--api-key dev_key \
--text "Parallel test run." \
--parallel 8 \
--out-dir outs \
--verbose