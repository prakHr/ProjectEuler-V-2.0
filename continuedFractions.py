def DivisorSumTwice401():
    LIMIT=10**15
    MOD=10**9
    splitCount=int(LIMIT**0.5)
    splitAt=LIMIT//(splitCount+1)
    #SIGMA=sum(FLOOR(N/i)*(i^2))
    #after splitat FLOOR(N/i) is always going to be 1
    #so directly found 
    def sumOfSquares(s,e):
        #(s+1)^2+(s+2)^2...+(e-1)^2+e^2
        return (e*(e+1)*(2*e+1)-s*(s+1)*(2*s+1))//6
    ans=0
    for i in range(1,splitAt+1):
        #i appears n//i times until splitAt 
        ans+=(i*i*(LIMIT//i))
    for i in range(1,splitCount+1):
        # For a given count of m = floor(n / k), which integer values of k yield this m?
        # m <= n/k, so mk <= n, and k <= n/m, thus k <= floor(n/m).
        # m > n/k - 1, so mk > n - k, and k(m + 1) > n, and k > n/(m+1), so k > floor(n/(m+1)).
        # floor(n / (m + 1)) < k <= floor(n / m)
        ans+=sumOfSquares(LIMIT//(i+1),LIMIT//i)*i
    return str(ans%MOD)
print(DivisorSumTwice401())
    
def isTerminatingDecimal(n,k):
    d=k
    while d%2==0:
        d=d//2
    while d%5==0:
        d=d//5
    
    if n%d:
        return n
    return -n

def D():
    #https://en.wikipedia.org/wiki/Repeating_decimal
    #(n/k)^k is maximized using differentiating it wrt k
    #so k=n/e
    #e itself is non terminating 2.718282818284...
    
    sum=0
    
        
    
    #print(k)
    #print(isTerminatingDecimal(11,k))
    for n in range(5,10001):
        k=round(n/2.718281828)
        sum+=isTerminatingDecimal(n,k)
    return sum
    
print(D())



def compute_207():
    #question of quadratic by putting n=2^t
    #it gives n*n-n-k=0
    #so k =(n)*(n-1)
    power2=set(2**i for i in range(20))
    count=0
    for a in range(1,2**19):
        if a+1 in power2:
            count+=1
        #now we want to run it till num/den<1/12345=>num*12345<den
        if count*12345<a:
            print('''P(%d)=%d/%d'''%(a*(a+1),count,a))
            break



def solve_138():
    result=0
    x=0
    y=-1
    for i in range(0,12):
        xnew=-9*x-4*y+4
        ynew=-20*x-9*y+8
        x=xnew
        y=ynew
        result+=abs(y)
    return result
print(solve_138())




def solve_94():
    x=2
    y=1
    limit=10**9
    result=0
    while True:
        aTimes3=2*x-1
        areaTimes3=y*(x-2)
        if(aTimes3>limit):
            break
        if(aTimes3>0 and areaTimes3>0 and aTimes3%3==0 and areaTimes3%3==0):
            a=aTimes3//3
            area=areaTimes3//3
            result+=3*a+1

        aTimes3=2*x+1
        areaTimes3=y*(x+2)
        
        if(aTimes3>0 and areaTimes3>0 and aTimes3%3==0 and areaTimes3%3==0):
            a=aTimes3//3
            area=areaTimes3//3
            result+=3*a-1

        nextx=2*x+3*y
        nexty=2*y+x

        x=nextx
        y=nexty
    return result
print(solve_94())

        

def solve_64():
    ans=0
    max=0
    for D in range(2,1001):
        x=int(D**0.5)
        if(x*x==D):
            continue
        
        m=0
        d=1
        a=x
        num=a
        den=1
        num1=1
        den1=0
        while(num*num-D*den*den!=1):
            m=a*d-m
            d=(D-m*m)//d
            a=(x+m)//d
            num2=num1
            num1=num
            
            den2=den1
            den1=den
            num=a*num1+num2
            den=a*den1+den2
            if(num>max):
                max=num
                ans=D
    return(ans)
print(solve_64())
            
            
