
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


