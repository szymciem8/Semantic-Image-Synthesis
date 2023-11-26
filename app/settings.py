API_PATH = 'http://api:8000/api/v1/{ml_type}/models/{model}/{action}/'
GAUGAN_PREDICT_URL = API_PATH.format(ml_type='generative', model='gaugan', action='predict')