# This is a python script

"""
Created on Sun Mar 26 11:32:01 2023

@author: szt
"""

def round_cy(number, num_place):
    
    import decimal
    # 修改舍入方式为四舍五入
    decimal.getcontext().rounding = 'ROUND_HALF_UP'
    
        
    n = len(str(number))-str(number).find('.')-1
    if n > num_place:
        for i in range(n,num_place,-1):
    
            precision = '%.'+str(i-1)+'f'
            precision = (precision % 0)
            number = decimal.Decimal(str(number)).quantize(decimal.Decimal(precision))
    
    elif n <= num_place:
        precision = '%.'+str(num_place)+'f'
        precision = (precision % 0)
        number= decimal.Decimal(str(number)).quantize(decimal.Decimal(precision))
        
    return number
