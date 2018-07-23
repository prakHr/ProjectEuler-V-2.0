#299(finding all possible triplets for 3 similar triangles)
def countIncenterCase(bound):
    res=0
    stack=[(0,1,1,1)]
    while stack:#(b-a)/(sqrt(2)x)=(sqrt(2)y/(d-c))=>(b-a)(d-c)=(2xy)
        a,b,c,d=stack.pop()#(a-b)(a-d)=2(a(a-x))
        n=a+c#-ad-ab+bd=aa-2ax
        m=b+d#Alternative solution @https://projecteuler.net/thread=299;page=3
        legsLength=m*m-n*n+2*m*n#2 cases x=a/2,b=d and b!=d
        if legsLength>=bound:
            continue
        if (m-n)%2!=0:
            res+=((bound-1)//legsLength)*2
        stack.append((a,b,n,m))
        stack.append((n,m,c,d))
    return res

def countParallelCase(bound):
    res=0
    stack=[(0,1,1,0)]
    while stack:
        a,b,c,d=stack.pop()
        n=a+c
        m=b+d
        legsLength=m*m+2*n*n+2*m*n
        if legsLength>=bound:
            continue
        if (m%2)!=0:
            res+=((bound-1)//(legsLength*2))
        stack.append((a,b,n,m))
        stack.append((n,m,c,d))
    return res

def solve299():
    bound=100000000
    x= countIncenterCase(bound)+countParallelCase(bound)
    return x
print(solve299())
    
#318(no of 9s in fractional part of (sqrt(p)+sqrt(q))^(2n))
import math
def N(p,q,n):
    rv=math.ceil(-n/math.log(math.sqrt(q)-math.sqrt(p),10))
    rv=(rv+rv%2)//2
    return rv
    
def solve318():
    rv,bound=0,2011
    for p in range(1,bound+1):
        for q in range(p+1,bound+1-p):
            if math.sqrt(q)-math.sqrt(p)<1:
                rv+=N(p,q,bound)
    return rv
print(solve318())
#309(Ladder Pythagoras theorem)
import itertools
import fractions

def intSqrt(n):#http://www.codecodex.com/wiki/Calculate_an_integer_square_root
    guess=(n>>n.bit_length()//2)+1
    res=(guess+n//guess)//2
    while abs(res-guess)>1:
        guess=res
        res=(guess+n//guess)//2
    while res*res>n:
        res-=1
    return res

def get(bound):
    wallsMap=[[] for _ in range(1+bound)]
    sqrtBound=intSqrt(bound)
    for n in range(1,sqrtBound+1):
        for m in range(1+n,sqrtBound,2):
            if n*n+m*m>=bound:
                break
            if fractions.gcd(m,n)>1:
                continue
            a=m*m-n*n
            b=2*m*n
            c=m*m+n*n
            for k in range(1,bound):
                if c*k>=bound:
                    break
                wallsMap[a*k].append(b*k)
                wallsMap[b*k].append(a*k)
    rv=0
    for walls in wallsMap:
        for a,b in itertools.combinations(walls,2):
            if (a*b)%(a+b)==0:#h=a*b/(a+b),where a=sqrt(x*x-w*w),b=sqrt(y*y-w*w)
                rv+=1
    return rv

def solve309():
    x=get(1000000)
    return x
print(solve309())
    
#326(Cycle modulo sums)

def getCyclePartialSums(m):
    cycle=[0,1]
    partialSum=0
    for i in range(2,6*m):
        partialSum=partialSum+(i-1)*cycle[i-1]
        cycle.append(partialSum%i)
    cycle=cycle[1:]+cycle[:1]#extra 0 which is at beginning is put at end
    cyclePartialSums=[cycle[0]]
    for i in range(1,6*m):
        x=(cyclePartialSums[-1]+cycle[i])%m
        cyclePartialSums.append(x)
    return cyclePartialSums

def getFrequencyMap(array,m):
    result=[0 for _ in range(m)]
    for i in array:
        result[i%m]+=1
    return result

def get(n,m):
    cyclePartialSums=getCyclePartialSums(m)#sequence has periodicity 6m
    cycleFrequencyMap=getFrequencyMap(cyclePartialSums,m)
    FrequencyMap=[0 for _ in range(m)]
    FrequencyMap[0]=1
    q,r=divmod(n,6*m)
    for i in range(r):
        FrequencyMap[cyclePartialSums[i]]+=1
    for i in range(m):#[(sigma i=p->q )ai mod m]=sigma[(i=0->p )ai modm]-sigma[(i=0->q )ai modm]
        FrequencyMap[i]+=(cycleFrequencyMap[i]*q)
    result=0
    for i in FrequencyMap:
        x=i*(i-1)//2
        result+=x
    return result
    
def solve326():
    for n,m in [(10,10),(10**12,10**6)]:
        print(n,m,'=>',get(n,m))
solve326()
#364(Alternative Questions occupying payphones in some relative order and toilet seats)
def solveExtendedGCD(a,b):
    x,y=0,1
    lastX,lastY=1,0
    while b!=0:
        q,r=divmod(a,b)
        a,b=b,r
        x,lastX=lastX-q*x,x
        y,lastY=lastY-q*y,y
    return lastX,lastY

def solveModInverse(a,m):
    x,y=solveExtendedGCD(a,m)
    return x%m

def setInverseFactorial(n):
    d=1
    mod=100000007
    result=[d]
    for i in range(1,1+n):
        d=(d*solveModInverse(i,mod))%mod
        result.append(d)
    return result

def setFactorial(n):
    d=1
    mod=100000007
    result=[d]
    for i in range(1,1+n):
        d=(d*i)%mod
        result.append(d)
    return result

def setPower2(n):
    d=1
    mod=100000007
    result=[d]
    for i in range(1,1+n):
        d=(d*2)%mod
        result.append(d)
    return result

def getOccupiedSeat(n,boundarySeat,factorial,power2,inverseFactorial):
    mod=100000007
    beginA=0 if (n-3+boundarySeat)%2==0 else 1
    endA=(n-3+boundarySeat)//3
    result=0
    for a in range(beginA,1+endA,2):#@ quasisphere
        b=(n-3+boundarySeat-3*a)//2#https://projecteuler.net/thread=364
        x=((factorial[a+b]**2)*factorial[a+b+1]*power2[a]*inverseFactorial[b])
        for i in range(1,2-boundarySeat+1):
            x*=(a+i)
        result+=x
    return result%(mod)

def get(n):
    mod=100000007
    factorial= setFactorial(n)
    power2=setPower2(n)
    inverseFactorial=setInverseFactorial(n)
    x=getOccupiedSeat(n,2,factorial,power2,inverseFactorial)
    y=getOccupiedSeat(n,1,factorial,power2,inverseFactorial)
    z=getOccupiedSeat(n,0,factorial,power2,inverseFactorial)
    return (x+2*y+z)%(mod)
    
def solve364():#448552540
    x=get(1000000)
    return x
print('=',solve364())
#202(LaserBeam)
def initPrimes( SIEVE_RANGE):
    primes=[]
    sieve_visited = [False] * SIEVE_RANGE
    sieve_visited[0] = sieve_visited[1] = True
    for i in range(2, SIEVE_RANGE):
        if sieve_visited[i] is False:
            primes.append(i)
            for j in range(i + i, SIEVE_RANGE, i):
                sieve_visited[j] = True
    return primes
                
def getPrimeFactors(n,primes):
    factors=[]
    d=n
    for i in range(len(primes)):
        if d==1:
            break
        p=primes[i]
        if d<p*p:
            break
        if d%p==0:
            while d%p==0:
                d=d//p
            factors.append(p)
    if d>1:
        factors.append(d)
    return factors

import itertools
def getFactorMultiples(factors):
    rv=[]
    n=len(factors)
    for i in range(1,n+1):
        for factorSubset in itertools.combinations(factors,i):
            x=1
            for j in factorSubset:
                x*=j
            rv.append([x,len(factorSubset)%2==1])
    return rv

def getMultiplyCount(n,multiply):
    firstK=None
    if multiply%3==1:
        firstK=(2*multiply-2)//3
    elif multiply%3==2:
        firstK=(multiply-2)//3
    lastK=(n-2)//3
    lastK=firstK+((lastK-firstK)//multiply)*multiply
    return(lastK-firstK)//multiply+1

def getWayCount(surfaceCount):#if we want 2n-3 reflections and n=1 mod 3 is divisible by r primes, each of which is 2 mod 3, then the number of ways is (phi(n) - 2^r)/3.
    primes=initPrimes(80000)
    n=(surfaceCount+3)//2
    factors=getPrimeFactors(n,primes)
    count=(n-2)//3+1
    for multiply in getFactorMultiples(factors):
        if multiply[1]:
            count-=getMultiplyCount(n,multiply[0])
        else:
            count+=getMultiplyCount(n,multiply[0])
    return count

def solve202():
    x=getWayCount(12017639147)
    return x
print(solve202())
#288(HyperExponentiation)

import sys
def totient(n):
    p=1
    i=2
    end=int(n**0.5)
    while i<=end:
        if n%i==0:
            p*=i-1
            n//=i
            while n%i==0:
                p*=i
                n//=i
            end=int(n**0.5)
        i+=1
    if n!=1:
        p*=n-1
    return p

def tetrationMod(x,y,m):
    if y==1:
        return x%m
    else:
        return pow(x,tetrationMod(x,y-1,totient(m)),m)
    
def solve288():
    x,y,m=1777,1855,10**8
    sys.setrecursionlimit(y+30)
    ans=tetrationMod(x,y,m)
    return str(ans)
print(solve288())
    
#243
import fractions
def isPrime(x):
    if x <= 1:return False
    elif x <= 3:return True
    elif x % 2 == 0:return False
    else:
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
def solve243():
    TARGET=fractions.Fraction(15499,94744)
    totient=1
    denominator=1
    p=2
    while True:#if d=p1^k1*p2^k2*...pm^km
        totient*=p-1#totient(d) = (p1-1) p1^(k1-1) * ... * (pm-1)pm^(km-1).
        denominator*=p
        while True:
            p+=1
            if isPrime(p):
                break
        if fractions.Fraction(totient,denominator)<TARGET:
            for i in range(1,p):
                numer=i*totient#using primorial(p)
                denom=i*denominator
                if fractions.Fraction(numer,denom-1)<TARGET:
                    return str(denom)
print(solve243())
     
#249(like 250)
def list_primality(n):
    result = [True] * (n + 1)
    result[0] = result[1] = False
    for i in range(int(n**0.5) + 1):
        if result[i]:
            for j in range(i * i, len(result), i):
                result[j] = False
    return result


def list_primes(n):
    return [i for (i, isprime) in enumerate(list_primality(n)) if isprime]


def solve249():#no of subsets of S(set of prime numbers)={2,3,5,...4999} sum of whose is a prime number
    LIMIT=5000
    MOD=10**16
    count=[0]*(LIMIT**2//2)
    count[0]=1
    s=0
    for p in list_primes(LIMIT):
        for i in reversed(range(s+1)):
            count[i+p]=(count[i+p]+count[i])%MOD
        s+=p
    isPrime=list_primality(s+1)
    ans=sum(count[i] for i in range(s+1) if isPrime[i])%MOD
    return str(ans)
print(solve249())
#250250
def solve250():
    MOD=10**16
    subsets=[0]*250
    subsets[0]=1
    for i in range(1,250250+1):
        offset=pow(i,i,250)
        subsets=[(val+subsets[(j-offset)%250])%MOD for (j,val) in enumerate(subsets)]
    ans=(subsets[0]-1)%MOD
    return str(ans)
print(solve250())
#255(Rounded Square Roots)
def RoundedSquareRootGet(n):#4.447401118025322
    d=len(str(n))
    xList=[2*10**((d-1)//2) if d%2==1 else 7*10**((d-1)//2)]
    while True:
        x=xList[-1]
        nextX=(x+(n+x-1)//x)//2#we are using floor instead of ceil
        if nextX==x:
            break
        else:
            xList.append(nextX)
    return len(xList)

def getMInterval(x,nInterval):
    return [(x+(n+x-1)//x)//2 for n in nInterval]

def getNInterval(x,m):#+n=(2m-1)x-xx+1,(2m+1)x-xx=n
    return [(2*m-1)*x-x**2+1,(2*m+1)*x-x**2]#(x+(n+x-1)//x)=2m or 2m+1=x+n/x

def get(d):
    iterationCount=0
    x0=2*10**((d-1)//2) if d%2==1 else 7*10**((d-1)//2)
    n0Interval=[10**(d-1),10**d-1]#initial interval
    initialState=(x0,n0Interval,1)
    dfsStack=[initialState]
    while dfsStack:
        x,nInterval,depth=dfsStack[-1]
        dfsStack.pop(-1)
        mInterval=getMInterval(x,nInterval)
        for m in range(mInterval[0],mInterval[1]+1):
            nextNInterval=getNInterval(x,m)
            nextNInterval[0]=max(nextNInterval[0],nInterval[0])
            nextNInterval[1]=min(nextNInterval[1],nInterval[1])
            nextState=(m,nextNInterval,depth+1)
            if x==m:
                iterationCount+=depth*(nextNInterval[1]-nextNInterval[0]+1)
                continue
            dfsStack.append(nextState)
    return iterationCount/(9*10**(d-1))
            
def solve255():
    x=get(14)
    return x
print(solve255())
#403(Lattice points b/w parabola and a line)
def L(a,b):
    c=int((a*a+4*b)**.5)#https://projecteuler.net/thread=403;page=2#last
    return int((c**3+5*c+6)/6)
#Lattice points are gonna be 1+axk+b-xk^2
#L(a,b)=(−x1+1−a−x1(1−x1)2+b(1−x1)−x1(1−x1)(1−2x1)6)+(x2+1+ax2(x2+1)2+b(x2+1)−x2(x2+1)(2x2+1)6)−(b+1)
#integrate A idea is that A consisting of L should be rational aa+4b=cc
def S(N):#x1=1/2(a-c),x2=1/2(a+c),b=aa-cc/4=>L(a,b)=L(-a,b)=L(c)
    midCutoff=int((N-(N-4)**.5*N**.5)//2)
    sqrtN=int(N**.5)
#Now use transformation u=x1,v=x2=>a=u+v,b=-uv,c=v-u
    z=0
    negUV=0
    s1=0
    s2=0
    s3=0

    #the u=0 axis that can be mirrored to the v=0 axis, minus the origin point
    for v in range(1,N+1):
        z+=L(0+v,-0*v)

    #the u=-v line which we subtract later due to doublecounting
    for u in range(1,sqrtN+1):
        v=-u
        negUV+=L(u+v,-u*v)

    #section I
    for u in range(-sqrtN,0):
        vmax=N//(-u)
        for v in range(-u,vmax+1):
            s1+=L(u+v,-u*v)

    #section II
    for u in range(1,midCutoff+1):
        vmax=N-u
        for v in range(u, vmax+1):
            s2+=L(u+v,-u*v)

    #section III
    for u in range(midCutoff+1,sqrtN+1):
        vmax=N//u
        for v in range(u, vmax+1):
            s3+=L(u+v,-u*v)

    #2*(zero axis + section I + section II + section III)-(u=-v line)+(L(0,0) origin point)
    return 2*(z+s1+s2+s3)-negUV+1

print (S(10**12))
#226(Area underCurve)
def s(x):
        n=int(x)
        return min(x-n,n+1-x)
def blancmangeCurve(x):
        y=0.0
        n=0
        epsilon=1e-15
        while True:
                dy=s(2**n*x)/2**n
                if dy<epsilon:
                        break
                y+=dy
                n+=1
        return y
def circle(x):#(x-0.25)^2+(y-0.5)^2=0.25^2
        return 0.5-(x*(0.5-x))**0.5#y-0.5=sqrt{0.25^2-(x-0.25)^2}=sqrt[x*(0.5-x)]
def findIntersection():
        p=0.0#using binary search
        q=0.25
        epsilon=1e-15
        while abs(p-q)>epsilon:
                r=(p+q)*0.5
                f=blancmangeCurve(p)-circle(p)
                g=blancmangeCurve(q)-circle(q)
                h=blancmangeCurve(r)-circle(r)
                if f*h<0.0:
                        q=r
                elif g*h<0.0:
                        p=r
        return (p+q)*0.5
def integrate(p,q,step):
        epsilon=1e-15
        sum=0.0
        x=p
        while x<=q:
                dy=blancmangeCurve(x)-circle(x)
                sum+=step*dy
                x+=step
        return sum
def solve226():
        epsilon=1e-15
        p=findIntersection()
        q=0.5
        return(integrate(p,q,1e-5))
print(solve226())
#279(integral sides with atleast one integral angle in a triangle)
def count90(bound):#Stern Brocot Tree
        res=0
        stack=[(0,1,1,1)]
        while stack:
                a,b,c,d=stack.pop()
                n=a+c
                m=b+d
                perimeter=2*m*(m+n)
                if perimeter>bound:continue
                if (m-n)%2!=0:
                        res+=(bound//perimeter)
                stack.append((a,b,n,m))
                stack.append((n,m,c,d))
        return res

def count60(bound):
        res=bound//3
        stack=[(0,1,1,1)]
        while stack:
                a,b,c,d=stack.pop()
                n=a+c
                m=b+d
                perimeter=(2*m+n)*(m+2*n)
                if perimeter>3*bound:
                        continue
                g=3 if (m-n)%3==0 else 1
                res+=((g*bound)//perimeter)
                stack.append((a,b,n,m))
                stack.append((n,m,c,d))
        return res

def count120(bound):
        res=0
        stack=[(0,1,1,1)]
        while stack:
                a,b,c,d=stack.pop()
                n=a+c
                m=b+d
                perimeter=(2*m+n)*(m+n)
                if perimeter>bound:continue
                if (m-n)%3!=0:
                        res+=(bound//perimeter)
                stack.append((a,b,n,m))
                stack.append((n,m,c,d))
        return res
def solve279():
        bound=10**8
        return count90(bound)+count60(bound)+count120(bound)
print(solve279())

#282(PizzaToplings f(m,n)) basically dirichlet convolution of f*g=f(n)Xg(n/d),where f is fact(md)/fact(d)**m,g is phi(n/d)
def f(m,n,factorial,divisor,eulerTotientFunction):#u(m,n)=fact(mn)/fact(n)**m-m*n/k*u(m,n/k)
        if n==1:#subtracting is done as rotation is not distinct and divide by (mn)
                return factorial[m-1]
        result=0
        nDivisor=divisor[n]
        for d in nDivisor:
                result+=factorial[d*m]//factorial[d]**m*eulerTotientFunction[n//d]
        result=result//(m*n)
        return result

def trivialCase(bound,factorial,divisor,eulerTotientFunction):
        res=0
        m=2
        while True:
                x=f(m,1,factorial,divisor,eulerTotientFunction)
                if x>bound:
                        break
                else:
                        res+=x
                        m+=1
        return res

def nonTrivialCase(bound,factorial,divisor,eulerTotientFunction):
        res=0
        m=2
        while True:
                n=2
                while True:
                        x=f(m,n,factorial,divisor,eulerTotientFunction)
                        if x>bound:
                                break
                        else:
                                res+=x
                                n+=1
                if n==2:break
                else:m+=1
        return res
        
def get(bound):
        bound1=100
        divisor=[[] for i in range(1+bound1)]
        for i in range(1,1+bound1):
                for j in range(i,1+bound1,i):
                        divisor[j].append(i)

        factorial=[1]
        for i in range(1,bound1+1):
                factorial.append(factorial[-1]*i)

        eulerTotientFunction=[i for i in range(1+bound1)]
        for i in range(2,1+bound1):
                if eulerTotientFunction[i]!=i:continue
                for j in range(i,bound1+1,i):
                        eulerTotientFunction[j]=eulerTotientFunction[j]-eulerTotientFunction[j]//i
        return trivialCase(bound,factorial,divisor,eulerTotientFunction)+nonTrivialCase(bound,factorial,divisor,eulerTotientFunction)           


print(get(10**15))
#283(sum(Perimeter) area/perimeter ratio=Z(RXR) b/w 1 to 1000)
import fractions
import itertools
def product(list):
        result=1
        for i in list:
                result*=i
        return result
def FactorisationGet(n,primes):
        d=n
        f={}
        for p in primes:
                if d==1 or p>d:
                        break
                e=0
                while d%p==0:
                        d=d//p#inputdata = [['a', 'b', 'c'],['d'],['e', 'f'],]
                        e+=1#result = list(itertools.product(*inputdata)),so result is gonna be a cartesian product of 3 list,taking 1 element from each list
                if e>0:
                        f[p]=e
        if d>1:f[d]=1
#result:
#[('a', 'd', 'e'),
 #('a', 'd', 'f'),
 #('b', 'd', 'e'),
 #('b', 'd', 'f'),
 #('c', 'd', 'e'),
 #('c', 'd', 'f')]
        unpacking=[[p**x for x in range(f[p]+1)]for p in f]#p is index in dictionary f ,with value of corresponding exponent
        return sorted([product(divisor) for divisor in itertools.product(*unpacking)])#divisor will be in form example 2^2,3^3,sounpacking=[[1,2,2^2],[1,3,3^2,3^3]]=>[1,2*1,3*1,2*3,3^2,2^2*3,2*3^2,3^3,2*3^3,2^2*3^3,2^2*3^3]
def hashTriangle(t):
        a,b,c=t
        return '''%d|%d|%d''' % (a, b, c)
def getCount(m,primes):
        #http://forumgeom.fau.edu/FG2007volume7/FG200718.pdf
        rv=0
        hashedTriangles=set()
        uList=FactorisationGet(2*m,primes)
        for u in uList:
                for v in range(1,int((3**0.5)*u)+1):
                        if fractions.gcd(u,v)!=1:
                                continue#1/2bcsinA/(a+b+c)=u,c=sqrt(a*a+b*b-2*a*bsqrt(1-sinA*sinA))=>sinA=1,a=2mu,b=2mv,c=a^2+b^2
                        #ratio=[mv*sqrt({(2mu)^2+(2mv)^2})/2mu+2mv+sqrt({(2mu)^2+(2mv)^2)}]
                        d=4*m**2*(u**2+v**2)#d={(2mu)^2+(2mv)^2}
                        deltaList=FactorisationGet(d,primes)
                        for delta1 in deltaList:
                                if delta1**2>d:
                                        break
                                delta2=d//delta1
                                q1,r1=divmod(delta1+2*m*u,v)#q1=(delta1+2mu)/v,q2=(delta2+2mu)/v,where delta1*delta2=d 
                                q2,r2=divmod(delta2+2*m*u,v)#q1+(2mv/u)=delta1+2mu/v+2mv/u,q2+(2mv/u)=delta2+2mu/v+2mv/u,q1+q2=(delta1+delta2+4mu)/v
                                if r1!=0 or r2!=0:
                                        continue
                                t=sorted([q1+(2*m*v)//u,q2+(2*m*v)//u,q1+q2])
                                hashedKey=hashTriangle(t)
                                if hashedKey not in hashedTriangles:
                                        hashedTriangles.add(hashedKey)
                                        rv+=sum(t)
                                       
        return rv

def problem283():
        primes=[]
        visited=[False]*(3000+1)
        visited[0]=True
        visited[1]=True
        for i in range(2,3000+1):
                if not visited[i]:
                        primes.append(i)
                for j in range(i+i,3000+1,i):
                        visited[j]=True
        bound=1000
        c=0
        for m in range(1,bound+1):
                c+=getCount(m,primes)
        return str(c)
if __name__=="__main__":
        print(problem283())

        
#290(sumofDigits(n)=sumofdigits(137n))
def getDigitSum(n):
        res=0
        d=n
        while d>0:
                res+=(d%10)
                d=d//10
        return res

def get(n):
        #SumDigits(10*n+d)= SumDigits(n)+d = SumDigits(137*(10*n+d))
        #SumDigits(137*(10*n+d))=SumDigits(137*n*10+137*d)=SumDigits(10*(137*n+(137*d)div10)+(137*d)mod10)=SumDigits(137*n+(137*d)div10)+(137*d)mod10
        #this is of the form SumDigits(137*(10*n+d))= SumDigits(137*n+x)+y,where x=digitCarry,y=digitSignature
        #If we set n=10*m+d (with m having one digit less than n)
        #then
        #SumDigits(137*n+x)+y = SumDigits(137*(10*m+d)+x)+y
        #= SumDigits(10*(137*m+(137*d+x) div 10) + (137*d+x) mod 10)+y= SumDigits(137*m+(137*d+x) div 10) + (137*d+x) mod 10+y
        #so a new x becomes  xnew=(137*d+x) div 10 and  ynew=(137*d+x) mod 10+y
        digitSignature=[(137*d)%10 for d in range(10)]
        digitCarry=[(137*d)//10 for d in range(10)]
        diffMap={0:{0:1}}
        for i in range(n):
                newDiffMap={}#we are gonna need a map of form {difference: (differenceValue,Carry)}
                for d in range(10):
                        for diff in diffMap:
                                for carry in diffMap[diff]:#(f2,p2)=((f1+137d1)div10,p1+d1-(f1+7*d1)mod10)
                                        x=carry+digitSignature[d]
                                        newSignature=x%10
                                        newDiff=((diff+d)-(newSignature))
                                        newCarry=digitCarry[d]+(x//10)
                                        if newDiff not in newDiffMap:
                                                newDiffMap[newDiff]={}
                                        if newCarry not in newDiffMap[newDiff]:
                                                newDiffMap[newDiff][newCarry]=0
                                        newDiffMap[newDiff][newCarry]+= diffMap[diff][carry]
                diffMap=newDiffMap#DP approach keeping track of carry and  diff going from k-1 digit numbers to k-digit numbers
        count=0
        for diff in diffMap:
                for carry in diffMap[diff]:
                        if diff==getDigitSum(carry):
                                count+=diffMap[diff][carry]
        return count
def solve290():
        count=get(18)
        return str(count)
print(solve290())

#323(based on Round-off Algorithm)
import fractions
import math
def binomial(n,k):
        return  math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
def solve323():#E[X0]=0,E[X1]=1-bit occuring occuring after 1 trial=1/2*(1+E[X1])+1/2*1=2
        SIZE=32#E[X2]=00+(01,10)+11=1/2*1/2*(1+E[X2])+1/2*(1/2+1/2)*(1+E[X1])+1/2*1/2*(1+E[X0])=8/3
        DECIMALS=10
        expect =[fractions.Fraction(0)] 
        for n in range(1,SIZE+1):#Generalizing for nth case E[XN]=sum(C(N,k)*1/(2^(K+N-K))*E[Xk] for k in range(N+1))
                temp=sum(binomial(n,k)*expect[k] for k in range(n))#E[XN]=(2^N+sum(C(N,k)*1/(2^(K+N-K))*E[Xk] for k in range(N)))/(2^N-1)
                expect.append((2**n+temp)/(2**n-1))#expect is gonna append it into format of Fraction
        ans=expect[-1]#finally we want E[X32]
        scaled=(ans)*(10**DECIMALS)
        whole=scaled.numerator//scaled.denominator
        frac=scaled-whole
        HALF=fractions.Fraction(1,2)
        if frac>HALF or (frac==HALF and whole%2==1):#rouding off algo ans=I+f,I will increase due to roundoff iff f>1/2 or (f=1/2 and I%2=0)
                whole+=1
        temp=str(whole)
        if DECIMALS==0:
                return temp
        temp=temp.zfill(DECIMALS+1)
        return( "{}.{}".format(temp[:-DECIMALS],temp[-DECIMALS:])  )
print(solve323())
#329(Prime Frog)
import fractions#probability computation can also be done using memo via recursion efficiently
def cancel(i, primes):
    for j in range(i*i, len(primes), i):
        primes[j] = False
 
def simpleSieve(hi):
    if hi == 2:
        return [2]
 
    mark = [False, False] + [True] * (hi-1)
    cancel(2, mark)
    #primes = [2]
    for i in range(3, hi, 2):
        if mark[i]:
            #primes.append(i)
            cancel(i, mark)
    return mark    

def solve235():
        START_NUM,END_NUM=1,500
        CROAK_SEQ = "PPPPNNPPPNPPNPN"
        assert 0<=START_NUM<END_NUM
        assert 1<=len(CROAK_SEQ)
        NUM_JUMPS=len(CROAK_SEQ)-1#initially positioned at one of the 500 squares and make a jump in left/right direction
        NUM_TRIALS=2**NUM_JUMPS#doing so requires 500*(2^14)
        globalNr=0
        limit=1000
        isPrime=simpleSieve(limit)#sieve
        for i in range(START_NUM,END_NUM+1):#For each starting square i,for each sequence of 2(jumpsPossibility) iterated with j
                for j in range(NUM_TRIALS):
                        pos=i#set initialPos and croak
                        trialNr=1
                        if isPrime[pos]==(CROAK_SEQ[0]=='P'):
                                trialNr*=2
                        for k in range(NUM_JUMPS):#simulate each jump and croak
                                if pos<=START_NUM:#forced moves
                                        pos+=1
                                elif pos>=END_NUM:
                                        pos-=1
                                elif (j>>k)&1==0:#Chosen Moves by checking if j is rightwise shifted wrt k,then they dont have any common bit with 0001 just like generating k-input truth table
                                        pos+=1
                                else:
                                        pos-=1
                                if isPrime[pos]==(CROAK_SEQ[k+1]=='P'):
                                        trialNr*=2
                        globalNr+=trialNr
        globalDr=(END_NUM+1-START_NUM)*(NUM_TRIALS)*(3**(len(CROAK_SEQ)))#Dr will consists of 3 from 2/3 and 3 from 1/3 for each fated event
        ans=fractions.Fraction(globalNr,globalDr)
        return str(ans)
if __name__=="__main__":
        print(solve235())#199740353/29386561536000
                                        
#230
A='''1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'''
B='''8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196'''
L=len(A)
fibonacci=[]
prev,curr=0,1
while curr<100000000000000000:#(127+19*n)*7**n at n=17
        fibonacci.append(curr)
        prev,curr=curr,prev+curr
        
def getWhichWord(n):
        n0=n-1
        for i in range(len(fibonacci)):
                if fibonacci[i]*L>n0:
                        return i
        return -1

def getDigit(n,whichWord):
        n0=n-1
        if whichWord==0:
                return A[n0]
        if whichWord==1:
                return B[n0]
        offset=(fibonacci[whichWord-2])*L
        if n0<offset:
                return getDigit(n,whichWord-2)
        else:return getDigit(n-offset,whichWord-1)
        return -1

def d(n):
        return getDigit(n,getWhichWord(n))


result=[d((127+19*n)*7**n) for n in range(18)]
print(''.join(result)[::-1])
#201 unique sum consisting of 50 elements from a set{1,4,9,16....100^2}
S=[n*n for n in range(1,101)]
T={}
for i in range(1+len(S)):
        T[i]={}
T[0][0]=1#initial subsetSize=0,subsetSum=0 with freq=1
for i in range(len(S)):
        U={}
        for subsetSize in T:
                if subsetSize>i or subsetSize>50:
                        continue
                for subsetSum in T[subsetSize]:
                        currentSum=subsetSum+S[i]
                        currentSize=subsetSize+1
                        if currentSize not in U:#using U[n][k][s+k^2] = T[n-1][k-1][s]+ U[n-1][k][s+k^2]
                                U[currentSize]={}#initiallizing frequency DP of u[size][subsetElementsSum]
                        if currentSum not in U[currentSize]:
                                U[currentSize][currentSum]=0
                        U[currentSize][currentSum]+=T[subsetSize][subsetSum]
        for subsetSize in U:
                for subsetSum in U[subsetSize]:
                        if subsetSum not in T[subsetSize]:
                                T[subsetSize][subsetSum]=0
                        T[subsetSize][subsetSum]+=U[subsetSize][subsetSum]
total=0
for subsetSum in T[50]:
        if T[50][subsetSum]==1:
                total+=subsetSum
print(total)
#209 τ(a, b, c, d, e, f) AND τ(b, c, d, e, f, a XOR (b AND c)) = 0
import itertools
def AssertCycle(cycle):
        n=len(cycle)
        for i in range(n):
                currentState=cycle[i]
                nextState=cycle[(i+1)%n]
                assert(nextState[0]==currentState[1])
                assert(nextState[1]==currentState[2])
                assert(nextState[2]==currentState[3])
                assert(nextState[3]==currentState[4])
                assert(nextState[4]==currentState[5])
                assert(nextState[5]==currentState[0]^(currentState[1]&currentState[2]))
def generateCycle():
        visitedInput=set()
        for input in itertools.product([0,1],repeat=6):#all possible 2^6=64 states are iterated over
                if str(input) in visitedInput:
                        continue
                currentState=input
                currentCycle=[]
                while str(currentState) not in visitedInput:#till bin(abcdef)<bin(bcdef(a^b&c))
                        visitedInput.add(str(currentState))
                        currentCycle.append(currentState)
                        a,b,c,d,e,f=currentState
                        currentState=(b,c,d,e,f,a^(b&c))
                AssertCycle(currentCycle)
                yield currentCycle
                
def getCount(cycleLength):
        zeroStartCount=1,0
        oneStartCount=0,1#we have to found the sequences starting with either 0 or 1 with no 2 consecutive 1s next to each other
        for i in range(cycleLength-1):#if 2 consecutive 1s are next to each other then t1ANDt2!=0
                zeroStartCount=zeroStartCount[0]+zeroStartCount[1],zeroStartCount[0]#S(n)=S(n-1)+S(n-2)
                oneStartCount=oneStartCount[0]+oneStartCount[1],oneStartCount[0]#T(n)=T(n-2)+T(n-1)
        return (zeroStartCount[0]+zeroStartCount[1]+oneStartCount[0])#because onrStartCount[1] will contain 11 at the end

def solve209():
        totalCount=1
        for cycle in generateCycle():
                count=getCount(len(cycle))
                totalCount=totalCount*count
        return str(totalCount)

print('209=>',solve209())
#215(Crack Free Walls)
def compute235():
        WIDTH,HEIGHT=32,10
        crackPositions=[]
        def getCrackPosition(cracks,position):
                if position<0:
                        raise ValueError()
                elif position<WIDTH:
                        for i in (2,3):
                                cracks.append(position+i)
                                getCrackPosition(cracks,position+i)
                                cracks.pop()
                elif position==WIDTH:
                        crackPositions.append(frozenset(cracks[:-1]))#frozenset is immutable set so element of set remains same after creation
                else:
                        return
        getCrackPosition([],0)
        nonCrackIndices=[[i for (i,cp1) in enumerate(crackPositions) if cp0.isdisjoint(cp1)] for cp0 in crackPositions]
        ways=[1]*len(crackPositions)
        for i in range(1,HEIGHT):
                newWays=[sum(ways[k] for k in nci) for nci in nonCrackIndices]#dynamin programming just like pascal triangle
                ways=newWays
        ans=sum(ways)
        return str(ans)
print(compute235())
#180(Fermats Little Theorem x^n+y^n-z^n=0 ,does not have a solution for abs(n)>2)
import itertools
import fractions
def initKeys(order):
        keys=set()#example is 12,13,14,23,24,34 for range(1,1+4)
        for a,b in itertools.combinations(range(1,order+1),2):#not taking the same pair
                keys.add(fractions.Fraction(a,b))#keys.add((a/b))
        keys=sorted(list(keys))
        return keys

def initNumbers(order):
        keys=initKeys(order)
        numbers={}
        for n in [-2,-1,1,2]:
                numbers[n]={}
                for key in keys:#so a hashmap of (n,keys**n)=>(keys)
                        numbers[n][key**n]=key
        return numbers

def getGoldenTriple(n,order):
        numbers=initNumbers(order)
        result=set()#example is 11,12,13,14,22,23,24,33,34,44 for range(1,1+4)
        for x,y in itertools.combinations_with_replacement(numbers[n],2):
                z=x+y
                if z in numbers[n]:#ifkey of z=key1+key2,where keys1,2 are of form key**n with value=key(a/b)
                        result.add(numbers[n][x]+numbers[n][y]+numbers[n][z])
        return result

def get(order):
        finalResult=set()
        for n in [-2,-1,1,2]:
                subResult=getGoldenTriple(n,order)
                finalResult=finalResult|subResult
        return sum(finalResult)
           
        
def compute180():
        result=get(35)
        print("answer =>", result.numerator + result.denominator)
        print("exact rational number =>", result)

if __name__=="__main__":
        print(compute180())
        
        
#61
import itertools
def figurateNum(sides,n):#https://oeis.org/wiki/Figurate_numbers
        return n*((sides-2)*n-(sides-4))//2

def findSolution(begin,current,sideUsed,sum,numbers):
        if sideUsed==0b111111000:#binary representation of 8
                if current%100==begin//100:#ordered cyclic nature
                        return sum
        else:
                for sides in range(4,9):
                        if(sideUsed>>sides)&1!=0:#in next recursive step sideused=sideused|(1<<4)=1000|(10000)=11000=24,11000>>4=00001 then sides=5,sideused=11000|1<<5=11000|100000=111000
                                continue
                        for num in numbers[sides][current%100]:#first 2 digit is obtained from begin and last 2 digits from current%100
                                temp=findSolution(begin,num,sideUsed | (1<<sides),sum+num,numbers)#when sideused=8 then all sides are used
                                if temp is not None:
                                        return temp
                return None
                        
def compute61():
        numbers=[[set() for j in range(100)]for i in range(9)]#we make a set of figurate numbers containing first 2 digits of 4 digit numbers as last 2 digit can be obtained by a different set of figurate
        for sides in range(3,9):
                for n in itertools.count(1):
                        num=figurateNum(sides,n)
                        if num>=10000:#reject not if exactly four digits
                                break
                        if num>=1000:
                                numbers[sides][num//100].add(num)#adding first 2 digits of num it to a particular set for P(3,n),...P(8,n)
        for i in range(10,100):
                for num in numbers[3][i]:
                        temp=findSolution(num,num,1<<3,num,numbers)#1<<3 is leftwise shift of 0001 to 1000
                        if temp is not None:
                                return str(temp)
        raise AssertionError("No solution")
if __name__=="__main__":
        print( compute61())
                                
#56
ans=max(sum(int(c) for c in str(a**b)) for  a in range(100) for b in range(100))
print(str(ans))
#88

def compute88():
	LIMIT = 12000
	minsumproduct = [None] * (LIMIT + 1)
	# Calculates all factorizations of the integer n >= 2 and updates smaller solutions into minSumProduct.
	# For example, 12 can be factorized as follows - and duplicates are eliminated by finding only non-increasing sequences of factors:
	# - 12 = 12. (1 term)
	# - 12 = 6 * 2 * 1 * 1 * 1 * 1 = 6 + 2 + 1 + 1 + 1 + 1. (6 terms)
	# - 12 = 4 * 3 * 1 * 1 * 1 * 1 * 1 = 4 + 3 + 1 + 1 + 1 + 1 + 1. (7 terms)
	# - 12 = 3 * 2 * 2 * 1 * 1 * 1 * 1 * 1 = 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1. (8 terms)
	def factorize(n, remain, maxfactor, sum, terms):
		if remain == 1:
			if sum > n:  # Without using factors of 1, the sum never exceeds the product
				raise AssertionError()
			terms += n - sum
			if terms <= LIMIT and (minsumproduct[terms] is None or n < minsumproduct[terms]):
				minsumproduct[terms] = n
		else:
			# Note: maxfactor <= remain
			for i in range(2, maxfactor + 1):
				if remain % i == 0:
					factor = i
					factorize(n, remain // factor, min(factor, maxfactor), sum + factor, terms + 1)
	
	for i in range(2, LIMIT * 2 + 1):
		factorize(i, i, i, 0, 0)
	
	# Eliminate duplicates and compute sum
	ans = sum(set(minsumproduct[2 : ]))
	return str(ans)


print(compute88())
    
#216(2*N*N)=0modp
def modpow(a,n,m):
    ret=1
    while n:
        if n&1:
            ret=(ret*a)%m
        n>>=1
        a=(a*a)%m
    return ret

def legendreSym(a,p):#p is odd prime
    ls=pow(a,(p-1)//2,p)#FERMATS LITTLE THM has a^phi(p)=a^(p-1)=1modp=>a^(p-1/2)=+/-1modp=>if it is congruent to -1modp,then it is not a quadratic residue of form (x^2)modp
    return -1 if ls==p-1 else ls#at a=(p-1)^2
    
def modular_sqrt(a,p):
    if legendreSym(a,p)!=1:return 0#a is not a square
    elif a==0:return 0
    elif p==2:return 1
    elif p%4==3:#a^(p-1/2)=1modp means a^((p+1)/2)=amodp means some(x=a^((p+1)/4)) setting p=4i+3(x=a^(i+1)) to get x^2modp
        return pow(a,(p+1)//4,p)
    start=p-1#after all this p-1 is gonna be of the form (start)*2^(end)
    end=0
    while start%2==0:#((p-1)/2) numbers are gonna be in our quadratic residue b/w {0,1,2,3...(p-1)}
        start//=2
        end+=1
    n=2#2^((p-1)/2)=1(modp)
    while legendreSym(n,p)!=-1:#we have to find a number n such that n^((p-1)/2)=-1(modp)
        n+=1
    x=pow(a,(start+1)//2,p)#first guess of square root
    b=pow(a,start,p)
    g=pow(n,start,p)
    r=end#end will decrease after each updation
    while True:
        t=b
        m=0#we are finding a least integer m such that b^(2^m)=1(modp) and 0<=m<=(r-1)
        for m in range(r):
            if t==1:
                break
            t=pow(t,2,p)
        if m==0:return x#if m=0 then we found our correct x otherwise update all x,b,g,r
        gs=pow(g,2**(r-m-1),p)#gs=g^(2^(r-m-1))
        g=(gs*gs)%p#gs=g^(2^(r-m-1)+2^(r-m-1))=g^(2^(r-m))
        x=(x*gs)%p
        b=(b*g)%p
        r=m
            
    
N=70710700#100sqrt(M  )
M=50*10**6
prime=[True]*N
prime_sq=[True]*M
prime[0]=prime[1]=False
for i in range(2,N):
    if i*i>N:
        break
    if not prime[i]:continue
    for j in range(i*i,N,i):
        prime[j]=False

for p in range(N):
    if not prime[p]:continue#From fermat little theorem a^(p-1)=1(modp)=>inv(a)modp=a^(p-2)
    inv=modpow(2,p-2,p)#2*N*N-1=0(modp) means 4*N*N=2(modp) means (2N)^2=2(modp) means inv(2)*((2N)^2)=1(modp) means 2^(p-2)=inv(modp)
    x=modular_sqrt(inv,p)#x^2(modp)=inv,where x=((N)^2)
    if x!=0:
        for i in range(x,M+1,p):#if x is non-quadratic residue then p-x is also non-quadratic in M*M skipping p-step 
            prime_sq[i-1]=False
        for i in range(p-x,M+1,p):
            prime_sq[i-1]=False
total=0
for i in range(2,M):
    if prime_sq[i] or (2*i*i-1<N and prime[2*i*i-1]):
        total+=1
print(total)
        
#150(changing shapes(min sub-Triangle sum) using pseudorandom Generator)
import sys
class MinimumSubTriangleProblem():
    def __init__self():
        self.triangle=None
        self.num_row=None
        self.acc_row_sums=None
    def min_sub_triangle(self,triangular_array):
        self._init_triangle(triangular_array)
        
        self._init_acc_row_sums()
        min_so_far=0
        for i in range(self.num_row):#i is iterator for row1,row2....rown
            for j in range(i+1):#j is iterator for each element in a row
                curr_sum=0
                for k in range(i, self.num_row):#for each element at j,triangle can be extended from rowi to rowi+1,...rown using iterator k
                    curr_sum+= self.acc_row_sums[k][(k-i)+j+1]- self.acc_row_sums[k][j]#triangle[x1...x2][y]=triangle[0...x2][y]-triangle[0...x1-1][y]
                    min_so_far = min(min_so_far, curr_sum)#example=s5+s8+s9=s3+s5+s8+s9-s3
        return min_so_far
    
    def _init_triangle(self,triangular_array):
        self.triangle=[]
        curr_row,curr_pos=1,0
        while curr_pos<len(triangular_array):
            self.triangle.append(triangular_array[curr_pos:curr_pos+curr_row])#[s1,[s2,s3],[s4,s5,s6]...]
            curr_row,curr_pos=curr_row+1,curr_pos+curr_row
        self.num_row=curr_row-1#0-based indexing 

    def _init_acc_row_sums(self):
        self.acc_row_sums=[]
        for row in self.triangle:
            acc_sum=0
            acc_row_sum=[0]
            for i in row:
                acc_sum+=i
                acc_row_sum.append(acc_sum)
            self.acc_row_sums.append(acc_row_sum)#[[[0],[s1]],[[0],[s2],[s2+s3]],[[0],[s4],[s4+s5],[s4+s5+s6]]...]
            
        
class LinearCongruentGenerator():
    def generate(self,bound):
        t=0
        for k in range(1,1+bound):
            t=(615949*t +797807) % 2**20
            yield t-2**19
            

def main():
    problem=MinimumSubTriangleProblem()
    generator=LinearCongruentGenerator()
    array=list(generator.generate(500500))
    print(problem.min_sub_triangle(array))
if __name__=='__main__':
    sys.exit(main())
#265(ans:209110240768)
#With n = 5, this means every candidate string must start with 000001 and end with 1.
# In other words, they are of the form 000001xxxxxxxxxxxxxxxxxxxxxxxxx1.
def check(digits,N):
    seen=set()
    digits=digits|(digits<<(2**N))#Leftwise shift of digits by 8=>0001xxx1=>0001xxx100000000
    #| will give us digits of form 0001xxx10001xxx1
    for i in range(2**N):
        seen.add((digits>>i)&(2**N-1))#for i=1  00001xxx10001xxx & 0111= 0000000000000xxx=xxx
    return len(seen)==2**N#for i=2 000001xxx10001xx & 0111=1xx,so it will keep us giving next 3 digits after that
def compute265():
    N=5
    twoPowN=2**N
    mask=2**N-1
    start=2**(2**N-N-1)+1#n=3 0001xxx1
    end=2**(2**N-N)#N=3,start=2**(4)+1=17 if xxx=000,end=2**(5)=32 if xxx=111
    ans=sum(x for x in range(start,end,2) if check(x,N))#17,19,21,23,25,27,29,31
    return str(ans)
print(compute265())
#137:nth Golden nugget=F(2N)*F(2N+1)
def F(N):
    sqrt5=5**0.5
    return (pow( (1+sqrt5)/2,N)- pow((1-sqrt5)/2,N) )/(sqrt5)
print(F(2*15)*F(2*15+1))#x/(1-x-xx)=n=>5nn+2n+1 is discriminant hence should be perfect square


#162
#   Note: (Have 0 or have 1 or have A) = (Have 0) + (Have 1) + (Have A) - (Have 0 and have 1) - (Have 0 and have A) - (Have 1 and have A) + (Have 0 and have 1 and have A).
#   Therefore:
#     Have 0 and have 1 and have A
#     = (15*16^(n-1) - 13^n) - (15*16^(n-1) - 15^n) - (15*16^(n-1) - 14*15^(n-1)) - (15*16^(n-1) - 14*15^(n-1)) + (15*16^(n-1) - 29*15^(n-1) + 14^n) + (15*16^(n-1) - 29*15^(n-1) + 14^n) + (15*16^(n-1) - 28*15^(n-1) + 13*14^(n-1))
#     = 15*16^(n-1) - 43*15^(n-1) + 41*14^(n-1) - 13^n.

def compute162():
    ans=sum(15*16**(n-1) - 43*15**(n-1) + 41*14**(n-1) - 13**n for n in range(1,17))
    return("{:X}".format(ans))#to compute the ans in hexagonal format
print(compute162())

#164
def  digitSum(n):
    return sum(int(c)for c in str(n))
def compute164():
    BASE,DIGITS,CONSECUTIVE,MAX_SUM=10,20,3,9
    innerLen=BASE**CONSECUTIVE
    ways=[[1]+[0]*(innerLen-1)]
    for digits in range(1,DIGITS+CONSECUTIVE+1):#upto 23 digits with padding 000 as well as lower ways consisr of padded 0s
        newRow=[]
        for prefix in range(innerLen):#from 000 to 999
            sum=0
            if digitSum(prefix)<=MAX_SUM:
                for nextDigit in range(BASE):
                    P=prefix%(BASE**(CONSECUTIVE-1))*BASE+nextDigit
                    sum+=ways[digits-1][P]
            newRow.append(sum)#[sum(ways[0][P]),sum(ways[0][P1])....] 
        ways.append(newRow)
    ans=ways[-1][0]-ways[-2][0]#ways[23][0]-ways[22][0]=000abcd...23-000defg...22=00(1-9 comes from a)esdf...20
    return str(ans)
print( compute164())
        
                    
                    
                    
            
#271(Very Fast ChineseThm)

def compute271():# 13082761331670030=product of FACTORS[i]
    FACTORS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
    factorSols=[[j for j in range(fact) if pow(j,3,fact)==1]for fact in FACTORS]
    def buildSol(i,x,mod):
        if i==len(FACTORS):
            return x
        else:
            fact=FACTORS[i]
            def Chinese(a,p,b,q):#solution is going to be of mixed-radix form x1+x2*p+x3*p*q for 2 primes p,q
                def reciprocal(x,m):#a=x1modp,b=(x1+x2*p)modq
                    assert 0<=x<=m#(b-x1)*inv(p)=x2
                    y=x#x2=(b-x1)*inv(p)%q,x1=a
                    x=m
                    a,b=0,1
                    while y!=0:
                        a,b=b,a-x//y*b
                        x,y=y,x%y
                    if x==1:
                        return a%m
                    else:raise ValueError("Reciprocal does not exist")   
                return (a+(b-a)*reciprocal(p%q,q)*p)%(p*q)#( x1+x2*p+x3*p*q)mod(pq)=(a+(b-a)inv(p,q or p%q,q)*p+0)mod(pq)
            return sum(buildSol(i+1,Chinese(x,mod,sol,fact),mod*fact) for sol in factorSols[i])
    ans=buildSol(0,0,1)-1
    return str(ans)
print(compute271())
#phi(n)=13!(248)
from math import log
def fact(x):
    return 1 if x==1 else x*fact(x-1)
fact13=fact(13)

def isPrime(n):
    if n == 2: return True
    if n % 2 == 0: return False
    if n == 6227020801: return False
    assert 1 < n < 4759123141 and n % 2 != 0, n
    s = 0
    d = n-1
    while d & 1 == 0:
        s += 1
        d >>= 1
    assert d % 2 != 0 and (n-1) == d*2**s
    for a in [2, 7, 61]:
        if not 2 <= a <= min(n-1, int(2*log(n)**2)):
            break
        if (pow(a, d, n) != 1 and all(pow(a, d*2**r, n) != (n-1) for r in range(s))):
            return False
    return True

factors=[]#combining these factors will give 13!
allFactors=[]
for p2 in range(0,11):
    for p3 in range(0,6):
        for p5 in range(0,3):
            for p7 in range(0,2):
                for p11 in range(0,2):
                    for p13 in range(0,2):
                         n = (2**p2)*(3**p3)*(5**p5)*(7**p7) *(11**p11)*(13**p13)
                         allFactors.append(n)
                         if isPrime(n+1) and n!=1:
                             factors.append(n)
cache={}
def f(minX,x):
    if x==1:return[[]]
    key=(minX,x)
    if key in cache:
        return cache[key]
    res=[]
    for i in factors:
        if i<=minX:
            continue
        if x%i==0:
            tmp=f(i,x//i)
            for l in tmp:
                res.append(l+[i])
    cache[key]=res
    return res

res=[]
for base in allFactors:
    tmp=f(0,base)
    for t1 in tmp:
        remainder=fact13//base
        v=1
        for t in t1:
            p=t+1
            k=1
            while remainder%p==0:
                remainder//=p
                k+=1
            v*=p**k
        if remainder==1:
            res.append(v)
            res.append(2*v)
            
        elif remainder%2==0:
            k=1
            while remainder%2==0:
                remainder//=2
                k+=1
            v*=2**k
            if remainder==1:
                res.append(v)
res.sort()
print(len(res),res[150000-1])
            


#Primonacci
def fibonacciMod(n,mod):#better method than fibonacci matrix
    a,b=0,1
    binary=bin(n)[2:]
    for bit in binary:
        a,b=a*(2*b-a),a*a+b*b
        if bit=="1":
            a,b=b,a+b
        a%=mod
        b%=mod
    return a


         
