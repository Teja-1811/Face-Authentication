import math
import random

def generate_otp(email):
    digits = "0123456789"
    otp =""
    
    for i in range(6):
        otp += digits[math.floor(random.random()*10)]
        #print(otp)
    
    return otp
        