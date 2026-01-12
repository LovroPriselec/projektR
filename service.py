from collections import deque
from predict import predict_next

def create_buffer(size):
    return deque(maxlen=size)

buffer = create_buffer(48)

#urediti čitanje iz csv da tu dolazi

def predictReq():
    if len(buffer)==48:
        return predict_next(list(buffer))
    else:
        return None