def SumOfDigits(n):
    sum=0
    while n!=0:
        sum+=n%10
        n=n//10
    return sum
x=0
for k in range(1,10**9):
    if k%23==0 and SumOfDigits(k)==23:
        x+=1
print(x)        
