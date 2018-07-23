


def get_sieve(until):
    sieve=[1]*until
    sieve[1]=0
    sieve[0]=0
    prime=2
    while prime<until:
        if sieve[prime]:
            temp=prime+prime
            while temp<until:
                sieve[temp]=0
                temp=temp+prime
        prime+=1
    return sieve


def get_primes_from_sieve(sieve):
    primes=[]
    for i in range(0,len(sieve)):
        if sieve[i]==1:
            primes.append(i)
    return primes



    
def solve_110():
    sieve=get_sieve(45)
    primes=get_primes_from_sieve(sieve)
    exponents=[0]*len(primes)
    result=1
    for i in range(0,len(primes)):
        result=result*primes[i]

    LIMIT=2*4000000-1
    count=1
    #1/x+1/y=1/n
    #let =n+r,y=n+s
    #n^(2)=r*s
    while(True):
        def Twos(LIMIT):
            exponents[0]=0
            divisors=1
            for i in range(0,len(exponents)):
                #where divisors number of divisors of N^(2),N expressed as p1^(a1)*p2^(a2)...
                #divisors=(2*a1+1)*(2*a2+1)....
                divisors=divisors*(2*exponents[i]+1)
                #calculate number of exponents of 2 2*(exponents[0]??)+1=LIMIT/divisors
                exponents[0]=int((LIMIT//divisors-1)//2)
            while divisors*(2*exponents[0]+1)<LIMIT:
                exponents[0]+=1


            
        Twos(LIMIT)
        #beneficial ans will come when exponents are arranged in nondecreasing order
        #i.e. a1>=a2>=a3...
        if exponents[0]<exponents[1]:
            count+=1
        else:
            def Number(primes,result):
                number=1
                for i in range(0,len(exponents)):
                    if exponents[i]==0:
                        break
                    #number N=p1^(a1)*p2^(a2)....
                    number=number*pow(primes[i],exponents[i])
                    
                return number
            
            number=Number(primes,result)
            if(number<result):
                result=number
            count=1
            
        if count>=len(exponents):
            break
         # Every time we have a candidate number where the number of twos are
        # lower than the number of threes we need to increment the exponent of
        # a larger prime factorand then set all smaller prime factors’s
        # exponents to the same number
        # this is not just applied for 2 and 3 but for all p1,p2,p3... in general
        exponents[count]+=1

        def SetAllSmallerExponents(count):
            for i in range(0,count):
                exponents[i]=exponents[count]
       
        SetAllSmallerExponents(count)
    return result
print(solve_110())
    
