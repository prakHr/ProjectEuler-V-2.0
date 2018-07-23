
#Z-algorithm(gives us the index according to prefix(from backward) substring comparisons with string in O(n) time)
s=('ababa')
N=len(s)
Z=[0]*(N)
Z[0]=N
L,R=0,0
for i in range(1,N):
    if i>R:
        L,R=i,i
        while(R<N and s[R-L]==s[R]):
            R=R+1
        Z[i]=R-L
        R=R-1
    else:
        j=i-L
        if Z[j]<R-(i-1):
            Z[i]=Z[j]
        else:
            L=i
            while(R<N and s[R-L]==s[R]):
                R=R+1
            Z[i]=R-L
            R=R-1

print(Z)
#Application 126B and Combinatorics
s=input()
n=len(s)
maxz,res=0,0
for i in range(1,n):
    if Z[i]==n-i and maxz>=n-i:
        res=n-i
        break
    maxz=max(maxz,Z[i])

#Chef and Divisors
def f(n):
    c=0
    l=[]
    try:
        while(n>1):
            if (n<limit and px[n]):
                l.append(2)
                break
            x=0
            while(n%prime[c]==0):
                n=n//prime[c]
                x+=1
            if(x!=0):
                l.append(1+x)
            c+=1
    except:
        l+=[2]
    return l

def g(l,p):
    if p==1:#if p=1 then it is leaf node in divisor tree
        return 0
    temp_l=l[:]
    temp_p=p#temp_p stores root values we dont have to subtract-1 for proper divisor as we are going to add +1 to consider the degree of non-leaf node of N
    pos=0
    for i in range(len(l)):
        if(l[pos]<l[i]):#p=p*(fmax+1-1)//(fmax+1)
            pos=i#index of value that contains max([a1+1,a2+1,a3+1,a4+1..])
    p=p//l[pos]#example consider 12=2^2*3^1=>p=(2+1)*(1+1)=6=>p=p//(2+1)=(1+1)=2
    l[pos]-=1#val=(2+1-1)=>p=p*val=(1+1)*(2)=4
    p=p*l[pos]
    return temp_p+g(l,p)
limit=10**6+1
px=[False]*2+[True]*(limit)
prime=[]
for i in range(limit):
    if(px[i]):
        prime.append(i)
        for j in range(i*i,limit+1,i):
            px[j]=False
            
a,b=map(int,input().split())
ans=0#N=p0^(a0)*p1^(a1)*p2^(a2)*p3^(a3)*p4^(a4)
print(f(12))
for n in range(a,b+1):
    l=f(n)#returns the list of all number of factors of prime divisors[a1+1,a2+1,a3+1,a4+1..] 
    p=1
    for e in l:
        p=p*e#p=total number of divisors=multiplication of l[elements]
    x=g(l,p)#x keep track of score of N
    ans+=x
    print(ans)
#Fill the matrix
def dfs(cur,color):
    visited[cur]=color
    for i in g[cur]:
        if visited[i]==0:
            dfs(i,color)

def solve():
    for (i,j) in v:
        x=i
        y=j
        if x==y:
            print('No')
            return
        elif visited[x]==0 and visited[y]==0:#case when both are uncolored
            dfs(x,BLACK)#then color x with black
            if visited[y]!=0:#in this process y should remain uncolored as it is going to be alternate 2-coloring
                print('No')
                return
            dfs(y,WHITE)
        elif visited[x]!=0 and visited[y]!=0:
            if visited[x]==visited[y]:#comes from case1 as they are disjoint sets
                print('no')
                return
        else:
            if visited[y]!=0:#this comes from case2 and similar to case1
                tmp=x
                x=y
                y=tmp
            if visited[x]==WHITE:
                dfs(y,BLACK)
            else:
                dfs(y,WHITE)
    print('yes')
    return

T=int(input())
BLACK,WHITE=1111,2222

for _ in range(T):
    n,k=map(int,input().split())
    v=[]
    visited=[0]*(1+n)
    g=[[] for i in range(1+n)]
    for i in range(k):
        x,y,z=map(int,input().split())
        if z==0:
            if x!=y:
                g[x].append(y)
                g[y].append(x)
        else:
            v.append((min(x,y),max(x,y)))
    v.sort()
    solve()
    
 

import sys
sys.setrecursionlimit(1000000)
def parser():
    while 1:
        data = list(input().split(' '))
        for number in data:
            if len(number) > 0:
                yield(number)

input_parser = parser()

def get_word():
    global input_parser
    return next(input_parser)

def get_number():
    data = get_word()
    try:
        return int(data)
    except ValueError:
        return float(data)

    for index in V:
        if node in E[index - 1]:
            E[index - 1].remove(node)


N = get_number()
M = get_number()

P = list()
order = dict()

for i in range(N):
    num = get_number()
    P.append(num)
    order[num] = i

E = [[] for i in range(N + 1)]

for i in range(M):
    a, b = get_number(), get_number()
    if (order[a] < order[b]):
        E[a].append(b)
    else :
        E[b].append(a)

for l in E:
    l.sort(key=lambda x: order[x])


# print(P)
# print(E)

it = iter(P)
node = next(it)

visited = [False for i in range(N + 1)]
visited[node] = True

def depth_first_search(node, it, P, E):
    res = True
    for neigh in E[node]:
        if res and not visited[neigh]:
            node = next(it, None)
            if node != neigh:
                return False
            visited[node] = True
            res = depth_first_search(node, it, P, E)
    return res

print(int(depth_first_search(node, it, P, E)))



#Max Quacks of duck
n,t=map(int,input().split())
arr=list(map(int,input().split()))
cnt,ans=[0]*1000100,[0]*(100010)
for i in arr:
    cnt[i]+=1
for i in range(1,1+t):
    if cnt[i]!=0:
        for j in range(i,1+t,i):
            ans[j]+=cnt[i]
ans1,ans2=0,0
ans1=max(ans1,ans)
for i in range(1,1+t):
    if ans[i]==ans1:
        ans2+=1
print(ans1,ans2)

import math
def sieve(n):
    
    is_prime = [True]*(n+1)
    prime = [2]
    
    for i in range(4,n+1,2):
        is_prime[i] = False
        
    for i in range(3,n+1,2):
        
        if(is_prime[i]):
            prime.append(i)
            
            for j in range(i*i,n+1,i):
                is_prime[j] = False
    return prime
n = int(input())
 
primes = sieve(n)
 
ar = [int(t) for t in input().split()]
pre = [0]
for i in range(1,n+1):
    pre.append(pre[i-1]+ar[i-1])
 
dp = [0]*(n+1)
 
for i in range(2,n+1):
    dp[i] = dp[i-1]
    
    for j in primes:
        
        if j>i:
            break
        
        p = i - j - 1
        
        if(p==-1):
            dp[i] = max(dp[i],pre[i])
        else:
            dp[i] = max(dp[i],pre[i]-pre[p+1]+dp[p])#pre[i]-pre[i-j]+dp[i-j-1] where j is prime
            
print(dp[-1])

#Timebank
T=int(input())
for _ in range(T):
    N,k=map(int,input().split())
    k+=1  
    A=list(map(int,input().split()))
    dp,mx=[0]*N,0
    if A[0]<0:dp[0]=0
    else:dp[0]=A[0]
    for i in range(1,N):
        if i-k>=0:
            dp[i]=max(dp[i-1],dp[i-k]+A[i])
        else:
            dp[i]=max(dp[i-1],A[i])
        if mx<dp[i]:
            mx=dp[i]
    print(mx)
#Ropes
T=int(input())
for _ in range(T):
    L=int(input())
    upperRope=list(map(int,input().split()))
    lowerRope=list(map(int,input().split()))
    time=L
    for i in range(1,L):
        time=max(time,upperRope[i-1]+i)
    for i in range(1,L):
        time=max(time,lowerRope[i-1]+i)
    print(time)
#CodeChef ALTARAY
T=int(input())
for _ in range(T):
    n=int(input())
    A=list(map(int,input().split()))
    dp=[1]*(n)
    for i in range(n-2,-1,-1):
        if A[i+1]* A[i]<0:
                dp[i]+=dp[i+1]
        
   
                  
    print(*dp)
                
                    
                    
            
import itertools
for  _ in range(int(input())):
    n,m=[int(i) for i in input().split()]
    l=[]
    for i in range(m):
        l.append([int(i) for i in input().split()])
    substitute=[0]*(m+1)#for subtask 2
    arr=[0]*(n+1)
    substitute[m]+=1
    ini=0
    for i in range(m,0,-1):
        ini+=substitute[i]
        ini%=10**9+7
        le=l[i-1][1]
        r=l[i-1][2]
        if l[i-1][0]==1:#subtask 1(Increase all elements by 1) 
            arr[le-1]+=ini
            arr[r]-=ini
            
        else:#subtask 2(2 l r 1<=l<=r<=m) Execute all commands whose indices are in range[l r]
            substitute[r]+=ini#in ini we have kept track of previous commands
            if le-1>=0:
                substitute[le-1]-=ini#As we want only those commands that are in range [l,r]
        #print(subs)
    #print(arr)
    mod=10**9+7
    arr[0]=arr[0]%mod
    for i in range(1,len(arr)):
        arr[i]=arr[i-1]+arr[i]
        arr[i]=arr[i]%mod
    
    #print(arr)
    print(*arr[:n])

#CodeChef TOTR

l=list(map(str,input().split()))
n=int(l[0])
s=list(l[1])
slen=len(s)
#print(n,s)
for _ in range(n):
    x=list(input())
    xlen=len(x)
    ans=''
    for i in range(len(x)):
        if (ord(x[i])-ord('a')>=0 and ord(x[i])-ord('a')<=25):
            ans+=s[ord(x[i])-ord('a')]
        elif(ord(x[i])-ord('A')>=0 and ord(x[i])-ord('A')<=25):
            ans+=s[ord(x[i])-ord('A')].title()#due to title there 
        elif(x[i]=='.' or x[i]=='!'):
            ans+=x[i]
        else:
            ans+=' '
    print(ans)
        
        
        
def reverse(x):
    l=len(x)
    for i in range(l):
        if x[i]!='0':
            y=i
            break
    for j in range(1,l+1):
        if x[-j]!='0':
            z=1-j
            break
    d=x[i:z+1]
    return d[::-1]
        

t = int(input())

for _ in range(t):
    x=input().split()
    n1=reverse(x[0])
    n2=reverse(x[1])
    r=str(int(n1)+int(n2))
    print(reverse(r))
#MUL(Karatsuba)
def multiply(x,y):
    if x.bit_length()<=y.bit_length():
        return x*y
    else:
        n=max(x.bit_length(),y.bit_length())
        half=(n+32)//64*32
        mask=(1<<half)-1
        xlow=x & mask
        ylow=y & mask
        xhigh=x>>half
        yhigh=y>>half
        a=mutiply(xhigh,yhigh)
        b=multiply(xlow+xhigh,ylow+yhigh)
        c=mutiply(xlow,ylow)
        d=b-a-c
        return(((a<<half)+d)<<half)+c
T=int(input())
for _ in range(T):
    x,y=map(int,input().split())
    print(multiply(x,y))
#POUR1
def gcd(a,b):
    while(b>0):
        temp=a%b
        a=b
        b=temp
    return a
def pour(A,B,C):
    a=A
    b=0
    c=1
    while(a!=C and b!=C):
        x=min(B-b,a)
        a=a-x
        b=b+x
        c=c+1
        if(a==C or b==C):
            break
        if(a==0):
            a=A
            c=c+1
        if(b==B):
            b=0
            c=c+1
    return c
T=int(input())
for _ in range(T):
    a=int(input())
    b=int(input())
    c=int(input())
    if c==a or c==b:
        print(1)
    elif c%gcd(a,b)!=0:
        print(-1)
    elif c>a and c>b:
        print(-1)
    else:
        x=min(pour(a,b,c),pour(b,a,c))
        print(x)
#CMPLS
def buildCache(seq,s,c):
    cache=[[0 for i in range(s+c)]for j in range(s+2)]
    for i in range(s):
        cache[0][i]=seq[i]
    for order in range(1,s):
        for idx in range(s-order):
            cache[order][idx]=cache[order-1][idx+1]-cache[order-1][idx]
    for i in range(1,1+c):
        cache[s-1][i]=cache[s-1][0]
    for order in range(s-2,-1,-1):
        for idx in range(s-order,s-order+c):
            cache[order][idx]=cache[order][idx-1]+cache[order+1][idx-1]
    print(cache[0][s:s+c+1])
T=int(input())
for _ in range(T):
    s,c=map(int,input().split())
    seq=list(map(int,input().split()))
    buildCache(seq,s,c)
#FCTRL
def Z(n):
    count=0
    while n>4:
        n=n//5
        count=count+n
    return count
T=int(input())
for _ in range(T):
    print(Z(int(input())))
#SPOJ Palin
def inc(left):
    leftList=list(left)
    last=len(left)-1
    while leftList[last]=='9':
        leftList[last]='0'
        last=last-1
    leftList[last]=str(int(leftList[last])+1)
    return("".join(leftList))

def nextPalindrome(num):
    size=len(num)
    odd=size%2
    if odd:
        center=num[size//2]
    else:
        center=''
    left=num[:size//2]
    right=left[::-1]
    pdrome=left+center+right
    if pdrome>num:
        print(pdrome)
    else:
        if center:
            if center<'9':
                center=str(int(center)+1)
                print(left+center+right)
                return
            else:
                center='0'
        if left==len(left)*'9':
            print('1'+(len(num)-1)*'0'+'1')
        else:
            left=inc(left)
            print(left+center+left[::-1])
T=int(input())
for _ in range(T):
    num=input()
    nextPalindrome(num)

for i in itertools.count():
    n=int(input())
    if n==42:
        break
    if n!=42:
        print(n)
   
	
def patterns(str,all):
    if len(str)==0:
        return all
    if str[0]=='0' or str[0]=='1':
        l=len(all)
        for i in range(0,l):
            all[i].append(str[0])
    if str[0]=='?':
        l=len(all)
        for i in range(0,l):
            temp=list(all[i])
            all.append(temp)
        for i in range(0,len(all)):
            if i<len(all)/2:
                all[i].append('0')
            else:
                all[i].append('1')
    return patterns(str[1:],all)
print(patterns('10?1?',[[]]))
