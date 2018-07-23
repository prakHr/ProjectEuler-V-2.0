
#Correcting Parenthesization
def enc(c,d):
    arr=['(','[','{',')',']','}']
    if(arr.index(c)//3>0 and arr.index(d)//3<1):return 2
    if(arr.index(c)%3==arr.index(d)%3 and c!=d):return 0
    return 1

def calc(i,j):
    if(i>j):return 0
    if(memo[i][j]>=0):return memo[i][j]
    r=calc(i+1,j-1)+enc(s[i],s[j])
    for k in range(i+1,j,2):
        r=min(r,calc(i,k)+calc(k+1,j))
    memo[i][j]=r
    return memo[i][j]

s=input()
n=len(s)
memo=[[-1]*n for i in range(n)]
print(calc(0,n-1))


def value(cNew,cOld):
    arr=['-','/','|','N']
    change=arr.index(cNew)-arr.index(cOld)
    if(change<0):change+=4
    return change

arr1=['S','L','F','R']
cOld,cNew=input(),input()
print(arr1[value(cNew,cOld)])

#998C
R=lambda:map(int,input().split())
#we have to calclate count of 1s before 0s
n,x,y=R()#agar x>y then ans=count*y,otherwise reverse all substrings but last (count-1)*x+y
s=input()
p=len(s.replace('1',' ').split())#replaces 1s with space and convert the zeros into list then find len
print([0,(p-1)*min(x,y)+y][p>0])


#998B
from itertools import*
R=lambda:map(int,input().split())
n,B=R()
a=[*R()]
c=[0,0]
r=[]
print(list(zip(a,a[1:])))#if lengths are not equal,dont worry
for x,y in zip(a,a[1:]):
    c[x%2]+=1
    if c[0]==c[1]:
        r+=[abs(x-y)]
        
r.sort()
print(next((i for i,x in enumerate(accumulate(r))if x>b),len(r)))#default is len(r)    
    
#Path Compression method on [l:r] segments
n=int(input())
points=[]

for i in range(n):
    l,r=map(int,input().split())
    points.append([l,1])
    points.append([r+1,-1])
points.sort()
ans=[0]*(1+n)
idx=0
for i in range(len(a)-1):
    idx+=a[i][1]
    ans[idx]=a[i+1][0]-a[i][0]
print(*ans)
    
    

#991C
def check(k,n):
    vasya=0
    curr=n
    while(curr>0):
        change=min(curr,k)
        vasya+=change
        curr-=change
        curr-=curr//10
    return 2*vasya>=n

candies=int(input())
for k in range(1,candies):
    if check(k,candies):
        exit(print(k))
    


#995C
def dist(a,b):
    return a*a+b*b

import random
n=int(input())
V = [tuple(map(int,input().split())) for i in range(n)]
indices=sorted((dist(a,b),i) for i,(a,b) in enumerate(V))

result=[0]*n
vx,vy=0,0
#first we compute model solution using sorting c1v1+c2v2+..
for d,i in reversed(indices):
    x,y=V[i]
    _,c=min(( ( dist(vx+x,vy+y),1 ),( dist(vx-x,vy-y),-1 ) ))
    vx+=c*x
    vy+=c*y
    result[i]=c
LIMIT=2.25*10**12
#we can randomly select it using while loop
while dist(vx,vy)>LIMIT: #then if vx,vy of v doesnt satisfy LIMIT
    i=random.randrange(n)#we randomly select a couple(x,y) from V and then instead of (vx+x,vy+y) value stored in that V
    c=result[i]
    x,y=V[i]#we store (vx+cx-cx-cx) i.e. vx-2cx(1cx to ractify +cx and other -cx to add x-cx)
    if dist(vx,vy)>dist(vx-2*c*x,vy-2*c*y):
        vx-=2*c*x
        vy-=2*c*y
        result[i]=-c
        
print(*result)

#996B
n=int(input())
a=list(map(int,input().split()))
i=0
ToAdd=0
while(True):
    if(i==n):ï=0  
    if(a[i]<=ToAdd):
        exit(print(i+1))
    i+=1
    ToAdd+=1
        
#996a
n=int(input())
a=0
for i in 100,20,10,5:
 a+=n//i
 n%=i
print(a+n)


#995B
n=int(input())
orderings=list(map(int,input().split()))
cnt=0#queue he
while orderings:
    i=orderings.index(orderings.pop(0))
    cnt+=i#har ek element ke liye uske corresponding nextSame[element] ka index add karenge
    del orderings[i]#aur uss element ko hata denge kyuki ye process greedily pehli element se chalo hogi
    


print(cnt)
        
        

#995A
n,k=map(int,input().split())
grid=[list(map(int,input().split())) for i in range(4)]

def it():
    for i in range(1,n):
        yield (1,i),(1,i-1),(0,i)#apan i=1 se dekhna suru karenge instead of i=0,similarly i=n-2 se instead of i=n-2
    yield(2,n-1),(1,n-1),(3,n-1)#agar car grid me 0 he toh answer hai
    for i in reversed(range(n-1)):#ya phir car parking lot ke samne he toh answer he
        yield (2,i),(2,i+1),(3,i)
    yield (1,0),(2,0),(0,0)
    
for (cx,cy),kuchBhi,(px,py) in it():
    if grid[cx][cy]==0 or grid[cx][cy]==grid[px][py]:break
else:exit(print(-1))

result=[]
parked=0
while parked<k:#(ans hamesa 20,000 se kam moves me hi aayega)
    for (cx,cy),(nx,ny),(px,py) in it():
        car=grid[cx][cy]
        if car==0:continue
        if grid[px][py]==car:#final location px,py me located he(ya toh car samne khadi hogi)
            result.append((car,px,py))
            grid[cx][cy]=0
            parked+=1
        elif grid[nx][ny]==0:#agar hamare pas answer he toh voh nx,ny se aayega(nahi toh car ko (nx,ny) me khada kerenge)
            result.append((car,nx,ny))
            grid[nx][ny]=car
            grid[cx][cy]=0
print(len(result))
for c,x,y in result:
    print(c,x+1,+1)
    
#976A
n,m,k=map(int,input().split())
if (k<n):print(1,k+1)
k-=n
rowfromBelow=k//(m-1)
rowFromAbove=(n-rowFromBelow)
columnFromRight=k%(m-1)
if(rowFromAbove & 1):
    ans=(m-columnFromRight)    
else:ans=(columnFromRight+2)
print(row,ans)
  
#983A
t=int(input())
for i in range(t):
    p,q,b=map(int,input().split())
    g=gcd(p,q)#(p/q)in base b=>0.abcd=>a*pow(b,-1)+b*pow(b,-2)=>q/[p*(pow(b,k))]=>apply gcd then q/b^k=>apply gcd
    q=q//g#p*q=gcd*lcm=>lcm/p=q/gcd
    b=gcd(q,b)
    while(b!=1):
        while(q%b==0):
            q=q//b
            b=gcd(q,b)
    if q==1:print("Finite")
            

#LIS by binary search
def ceilIndex(arr,T,end,s):
    start=0
    Lenn=end
    while(start<=end):
        mid=(start+end)//2
        if(mid<Lenn and arr[T[mid]]<s and s<=arr[T[mid+1]]):
            return mid+1
        elif arr[T[mid]]<s:
            start=mid+1
        else:end=mid-1
    return -1

def LongestIncreasingSub(arr,x):#due to nature of binary search picking lexicographically largest by
    #indices and also smallest possible sum 
    
    R=[-1]*(x)
    T=[0]*(x)
    Len=0
    for i in range(1,x):
        
        if arr[T[0]]>arr[i]:
            T[0]=i
        elif arr[T[Len]]<arr[i]:
            Len+=1
            T[Len]=i
            R[T[Len]]=T[Len-1]#the last value of increasing subsequence should be minimum
        else:
            index=ceilIndex(arr,T,Len,arr[i])
            T[index]=i
            R[T[index]]=T[index-1]
    
    index=T[Len]
    ans=[]
    while(index!=-1):
        ans.append(arr[index])
        index=R[index]
    print(ans)
    return Len+1
            
arr=[3,4,-1,5,8,2,3,12,7,9,10]
x=len(arr)
print(LongestIncreasingSub(arr,x))
    

#940C
n,k=map(int,input().split())
s=input()
s=sorted(set(s))
if k>n:
    print(s+s1[0]*(b-a))
else:
    i=k-1
    while s[i]>=s1[-1] and i>-1:i-=1
    d=s1.index(s[i])
    print(s[:i]+s1[d+1]+s1[0]*(k-i-1))
    

#920c
n=int(input())
m=0
a=[int(x) for x in input().split()]
s=input()
m=0
for i in range(n-1):
    m=max(m,a[i])
    if m>i+1 and s[i]=='0':
        exit(print('NO'))

#772B(Obtuse angle for min distance for nonconvexity)
n=int(input())
fucker=lambda i,j: (t[i][0]-t[j%n][0])**2+(t[i][1]-t[j%n][1])**2
t=[list(map(int,input().split())) for _ in range(n)]
h=1e20
for i in range(n):
    a,b,c=fucker(i-1,i),d(i,i+1),d(i-1,i+1)
    h=min(h,(4*a*b-(a+b-c)**2)/c)#Heron's formula from wikipedia
print(h**0.5/4)
#801A
s='K'+input()+'V'#for border padding since we are looking for vk if it appears right at the beginning we pad it with k and same logic for end
print(s.count('VK')+('VVV' in s or 'KKK' in s))



        
#772A
def someSearch(time,a,exhausts,P):
    ans=0
    for i in range(n):
        ans+=max(0,(time-exhausts[i])*a[i])
    if time*P>=ans:return 1
    return 0

n,P=map(int,input().split())
a,b,exhausts=[],[],[]
for i in range(n):
    x,y=map(int,input().split())
    a.append(x)
    b.append(y)
    exhausts.append(y/x)
flag=0
if sum(a)<=P:
    flag=1
    print('-1')
if flag==0:
    l,r=0,10**18
    for i in range(220):
        mid=(l+r)/2
        if (someSearch(mid,a,exhausts,P)):
            l=mid
        else:
            r=mid
    if r>10**18-100:
        print(-1)
    else:print('{0:.9f}'.format(l))
#817C
def sumofDigits(num):
    ans=0
    while num>0:
        ans+=(num%10)
        num=num//10
    return ans
        

    
n,s=map(int,input().split())
l,r=0,n
while l<r:
    mid=(l+r)//2
    if (mid-sumofDigits(mid)<s):
        l=mid+1
    else:
        r=mid
flag=1
if n==l:flag=0
print(n-l+flag)

   
    
#How to rotate sides top,front,bottom,back,left,right
def apply(y,x):
    ans=[0]*6
    for i in range(6):
        ans[i]=y[x[i]]
    return ans

pos,L,F=[],[3,0,1,2,4,5],[5,1,4,3,0,2]
for i in range(4):
    for j in range(4):
        for k in range(4):
            res=list(range(6))
            for x in range(i):res=apply(res,L)
            for x in range(j):res=apply(res,F)
            for x in range(k):res=apply(res,L)
            if res not in pos:
                pos.append(res)
#for i in range(24):print(pos[i])
#989B
n,p=map(int,input().split())
s=input()
x=s.replace('.','0')
t=[int(i) for i in x]
if all(t[i]==t[i+p] for i in range(n-p)):#since ans will always be there if p<(n/2) or 2*p<n
    x=s[:n-p:].find('.')#first n-p elements
    y=s[p::].find('.')#after p remaining elements
    if x!=-1:#then we can find corresponding 1 or 0 for x or y+p
        t[x]=t[x+p]^1
    elif y!=-1:
        t[y+p]=t[y]^1
    else:print("No")
print(*t)
    
    


#75B

r={'posted':15,'commented':10,'likes':5}
urName=input()
a={}
m=set()
for _ in '0'*int(input()):
    t=input().split()
    s=t[0]
    p=t[3-(t[1]=='likes')][:-2]#we dont want 's
    m.add(s)
    m.add(p)
    if s==urName:
        a[p]=a.get(p,0)+r[t[1]]
    if p==urName:
        a[s]=a.get(s,0)+r[t[1]]
if urName in m:
    m.remove(urName)
for v in sorted(set(a.values()) | set([0]))[::-1]:#exclude ur name from even if a[Name]=0
    print('\n'.join(sorted(s for s in m if a.get(s,0)==v)))
    

    
#356A#we have to use path compression techniques
n,m=map(int,input().split())
path,d=[0]*(n+2),list(range(1,n+3))
for i in range(m):
    li,ri,xi=map(int,input().split())
    while li<xi:
        if path[li]:
           tmp=d[li]#swap(li,d[li]) and d[li]=xi
           d[li]=xi
           li=tmp
        else:
            d[li],path[li]=xi,xi
            li+=1
    li+=1
    ri+=1
    while path[ri]:
        ri=d[ri]
        
    while li<ri:
        if path[li]:
           tmp=d[li]
           d[li]=ri
           li=tmp
        else:
            d[li],path[li]=ri,xi
            li+=1
print(' '.join(map(str,path[1:-1])))
        
    
   
   


#910C
b=['a','b','c','d','e','f','g','h','i','j']
l=[0]*20
z=[]
n=int(input())
for i in range(n):
    s=input()
    z.append(s[0])#z consists of letters with MSB i.e 10**(len of longest string) as we do nt want leading digit to be 0
    for j in range(len(s)):
        l[b.index(s[j])]+=pow(10,(len(s)-j-1))#suppose string is 'aXaXXX'+'axa'=>thenl[0]=10**(5)+10**(3)+10**(2)+1
ans,h,j=0,0,1
for k in range(10):#code will iterate according to range(len(b)) picking maxx corresponding to each letter in b greedily checking if its not a leading digit and calculating our ans by adding maximum powers of 10s by using lower (j)s
    maxx=0
    for i in range(10):#now we have to find maxx=>corresponds to maximum powers of 10s stored corresponding to b[]
        if l[i]>maxx:
            maxx=l[i]
    bi=b[l.index(maxx)]#we dont want b to be a leading digit
    l[l.index(maxx)]=0#we will make digit=0 corresponding to the maximum powers of 10s 
    if not (bi in z) and h==0:#if ans with maxx is not the one with the longest length string character and its not a leading digit 
        h=1#then we plant a flag named h on it
    else:
        ans+=maxx*j
        j+=1
print(ans)

#919C
import re
n, m, k = map(int, input().split())
s=[input() for _ in range(n)]
if n>1 and k>1:
    t=list(map(lambda x:"".join(x),zip(*s)))#code to extract vertical combination of **,..,*.,.*
    s+=t#now s will contain horizontal and vertical consecutive positions 
ans=0
p=re.compile(r"\.{"+str(k)+",}")#re.compile('\\.{2,}')

for i in s:
    #len comes from x=(p.findall(i))
    ans+=sum(map(lambda x: len(x)-k+1,p.findall(i)))#ans =len-k+1
print(ans)

        


        

table=set([('.#.#..#.'),('.....#..'),('.#.#..#.'),('#.#....#'),('.....#..')])
print(table)
print(list( zip(*table)))
#924A
n,m=input().split()
table=set(input() for _ in [0]*int(n))

print(['No','Yes'][all(sum(hashtag<'.' for hashtag in column)<2 for column in zip(*table))])#checking for all rows and total number of rows with sum(hashtag)<2

#416C
n=int(input())
Tables=[]
for i in range(n):
    size,money=map(int,input().split())
    Tables+=[(money,size,i)]
k=int(input())
maximumPeopleOnATable=list(map(int,input().split()))
s,q=0,[]
for (money,size,index) in reversed(sorted(Tables)):
    jMin,peopleMax=-1,10000
    for (j,people) in enumerate(maximumPeopleOnATable):
        if size<=people<peopleMax:
            jMin=j
            peopleMax=people
    if jMin>-1:
        maximumPeopleOnATable[jMin]=0
        q+=[(index,jMin)]
        s+=money
print(len(q),s)
for (i,j) in q:
    print(i+1,j+1)
            
    


#416B(Painters spending time on a particular painting before moving to next painting)
m,n=input().split()
m=int(m)
n=int(n)
a=[0]*8
for i in range(m):
    tij=list(map,int,input().split())
    for j in range(1,1+n):
        a[j]=max(a[j],a[j-1])+tij[j-1]#to make sure next painter does nt start working on it earlier
    print(a[n],end=' ')
    
#988C
u=[]
for i in range(int(input())):
    n=int(input())
    a=list(map(int,input().split()))
    s=sum(a)
    u+=[(s-a[j],(i+1,j+1)) for j in range(n)]
u=sorted(u)
for i in range(len(u)-1):
    if u[i][0]==u[i+1][0] and u[i][1][0]!=u[i+1][1][0]:
        print('YES');print(u[i][1][0],u[i][1][1]);print(u[i+1][1][0],u[i+1][1][1])
print("NO")

#988B
a=sorted([input() for i in range(int(input()))],key=len)
print('YES',*a,sep='\n') if all(x in y for x,y in zip(a,a[1:])) else print('NO')

#988A
n,k=map(int,input().split())
arr1=list(map(int,input().split()))

arr=set(arr1)
if len(arr)<k:
    print("NO")
else:
    print("YES")
    freq=[0]*(100)
    for i in arr1:
        freq[i]+=1
    for i in range(len(arr1)):
        if freq[arr1[i]]>0:
            freq[arr1[i]]=0
            print(i+1,end=' ')
    
    
            

            
   
#938C(Constructing tests)
def calc(xi):#since n*n-(n//m)*(n//m)=x=>n=sqrt(m*m*x/(m*m-1)) for m>2 and m=n sqrt(x+1)
    n=1+int(pow(xi,0.5))
    n2=pow(n,2)#here (1+xi**0.5)**2=1+xi+2*(xi**0.5)<=2*xi
    while n2<=2*xi and n<=10**9:
        nsm2=n2-xi#we have to choose different values of columns
        nsm=pow(nsm2,0.5)
        if nsm==int(nsm):
            m=n//nsm
            if n2-pow(n//m,2)==xi:#here n//m is number of nonintersecting sbmatrices in nXm
                return n,int(m)#so n*n-(n//m)*(n//m) is number of maximum 1s in matrix
        n+=1
        n2=pow(n,2)
    return -1,-1
t=int(input())
for _ in range(t):
    xi=int(input())
    if xi==0:
        print("1 1")
        continue
    n,m=calc(xi)
    if m==-1:
        print(-1)
    else:print(n,m)
#939C(Convenient for everybody)
n=int(input())
arr=list(map(int,input().split()))
s,f=(map(int,input().split()))
delta=f-s
first=s
mx=m=sum(arr[:delta])
for i in range(n):
    m=m-arr[i]+arr[(i+delta)%n]#taking 1+2,1+2+3-1=2+3 for delta=3 for 1 2 3
    if m==mx:
        first=min(first,(n-i-2+s)%n+1)
    if m>mx:
        mx=m
        first=((n-i-2+s)%n+1)#time of beginning of contest (inTZ1) 1 2 3 4|1 2 3 4 ,(inTZ2) 2 3 4|1 2 3 4...
print(first)
#466C(Split array into 3 equal sum parts)
n=int(input())
arr=list(map(int,input().split()))
ans=0
s=sum(k)
if s%3==0:
    first=s//3
    second=s-first
    s=t=0
    for i in range(n-1):
        s+=arr[i]
        if s==second:ans+=t
        if s==first:t+=1
print(ans)

#919B(PerfectNumber)
def y(first):
    t=0
    while first:
        t=t+(first%10)
        first=first//10
    return t
n=int(input())
first=19
while n>1:
    first=first+9
    if y(first)==10:
        n=n-1
print(first)

#294C
N,M=2000,10**9+7
f=[1]*N
for i in range(1,N):
    f[i]=(i*f[i-1])%M
n,m=map(int,input().split())
ar=sorted(map(int,input().split()))
b=[]
for i in range(1,m):
    d=ar[i]-ar[i-1]-1
    if d>0:
        b.append(d)
res=(pow(2,sum(b)-len(b),M)*(f[n-m]))%M
b=[ar[0]-1]+b+[n-ar[-1]]#dividing into types ...(A)#...(B)#....(C)
for br in b:#then taking inverse factorial of these types
    res=res*pow(f[br],M-2,M)%M
print(res)
#985D
def check(H,n,h):
    k=min(h,H)#the leftmost pillar will have height 
    cnt=H*H-(k*(k-1)//2)#total number of sand packs required to build 
    return n>=cnt#will have the minimal width you can achieve.

def get(H,n,h):
    k=min(h,H)
    cnt=H*H-(k*(k-1)//2)
    lenX=(H+(H-1))-(k-1)# the width of resulting truncated pyramid which is H+(H-k)
    return lenX+((n+H-cnt-1)//H)#plus the minimal number of additional pillars it will take to distribute leftover sand packs
    
n,h=map(int,input().split())
l,r=1,2*(10**9)
while l+1<r:
    mid=(l+r)//2
    if check(mid,n,h):
        l=mid
    else:
        r=mid
if check(r,n,h):
    print(get(r,n,h))
print(get(l,n,h))
#478B
n,k=map(int,input().split())
nk=n//k#(nk(nk-1)-nk(nk+1))=nk(nk-1-nk-1)=-2nk,So basically if nk!=0 kmn=(n/m*(n/m-1))/2
#kmn=kmn*(m-(n%m)) then kmn+=lft*(n%m) where lft=(n/m*(n/m+1))/2
print((nk*(nk-1))//2*k+n%k*nk,(n-k+1)*(n-k)//2)

#707C
n=int(input())
if n<3:
    print("-1")
elif n*n%4:
    print(n*n//2,n*n//2+1)
else:
    print(n*n//4-1,n*n//4+1)
    
#230B(TPrimes are Squares of Prime Numbers)
n=2**20
a=[1]*n
r={4}
for i in range(2,n):
    if a[i]:
        for  j in range(i*i,n,i):
            a[j]=0
        r.add(i*i)
input()
for d in map(int,input().split()):
    print(['NO','YES'][d in r])
    
#659C(Toys of distinct types)
I=lambda:map(int,input().split())
n,m=I()
a=set(I())
b=[]
i=1
while i<=m:
    if i not in a:
        b.append(str(i))
        m-=i
    i+=1
print(len(b),'\n'+' '.join(b))    

#451B(Sort the Array)
n=int(input())
a=[int(x) for x in input().split()]
b=sorted(a,key=int)
l,r=0,n-1
if a==b:
    print("yes")
    print("1 1")
else:
    while a[l]==b[l]:
        l+=1
    while a[r]==b[r]:
        r-=1
    if a[l:r+1]==b[l:r+1][::-1]:#happens because only 1 segment sweep is required
        print("yes")
        print(1+min(l,r),1+max(l,r))
    else:
        print("no")
        
#538B(Quasi Binary)
n=input().strip()
m=max([int(i) for i in n])
print(m)
for i in range(m):
    print(int(''.join(['1' if int(j) > i else '0' for j in n])),end=' ')

#581B
n=int(input())
CurrentFloors=list(map(int,input().split()))
ans=[]
ans.append(0)
maxValue=CurrentFloors[-1]
for i in range(2,1+n):
    maxValue=max(maxValue,CurrentFloors[-i+1])
    ans.append(maxValue-CurrentFloors[-i]+1)


#158B
n=input()
a,b,c,d=map(input().count,('1','2','3','4'))
print(d+c+(2*(b+1)+max(0,a-c)+1)//4)
  

#509C
t=[0]*400
x=k=0
for q in range(int(input())):
    y=int(input())
    d,x=y-x,y#x consists of b[i-1] and y=b[i],d=3-0=3,x=3
    j=0#j=0
    while d<1 or t[j]>min(8,9*(j+1)-d):#t[0]=0>6
        d+=t[j]
        t[j]=0
        j+=1
    t[j]+=1
    k=max(k,j)
    a,b=divmod(d-1,9)
    t[:a]=[9]*a
    t[a]+=b#t[0]=2+1
    print(''.join(map(str,t[k::-1])))

#955B
from collections import Counter
d = Counter(input())

if sum(d.values()) < 4 or len(d) > 4 or len(d) == 1:
    print('No')
elif len(d) >= 3:
    print('Yes')
elif any(d[k] == 1 for k in d):
    print('No')
else:
    print('Yes')
    
    
#Chef and Cycled Cycles
T=int(input())
for _ in range(T):
    n,q=map(int,input().split())
    graph=[]
    sizes=[]
    connections=[]
    junctions=[]
    query=[]
    for j in range(n):
        temp_array=[int(i) for i in input().split()]
        sizes.append(temp_array[0])#sizes Takes no of nodes in different cycles
        for elem in range(2,sizes[j]+1):
            temp_array[elem]+=temp_array[elem-1]#we have take -2 due to this as we want 0-based indexing
        graph.append(temp_array[1:])#graph contains Prefix array sum of cycles array
    for j in range(n):
        temp_array=[int(i) for i in input().split()]
        junctions.append([temp_array[0],temp_array[1]])
        connections.append(temp_array[2])
    for j in range(1,n):
        connections[j]+=connections[j-1]#sums of previous bridges weights in pair of cycles Ci,Ci+1
    for j in range(q):
        temp_array=[int(i) for i in input().split()]
        query.append(temp_array)
    beg=[]
    end=[]
    new_junction=[]
    new_junction.append([junctions[-1][1],junctions[0][0]])
    for j in range(n-1):
        new_junction.append([junctions[j][1],juctions[j+1][0]])
    for j in range(n):
        s,e=new_junction[j]#so our starting edge node is the node of the bridge coming from last junction to first
        temp_array=[]#our ending edge node is the node of the bridge coming from second junction to first
        if sizes[j]>=2:
            for z in range(1,sizes[j]+1):
                fixed=s
                if fixed==z:
                    temp_array.append(0)
                    continue
                if z==1:#here we are not handing it wrt starting node so we have to later on add foo and bar
                    m1=graph[j][fixed-2]#2 nodes back from fixed node as size=2 if z is 1st node(-2 is due to -1(elem)+-1(0-based indexing))
                    m2=graph[j][-1]-m1#so we have 2 ways now ,another one is total prefix sum-first way,stored in last position of graph list
                    temp_array.append(min(m1,m2))
                    continue
                else:#if z is not first node and not starting node then we have to store wrt to (z-fixed)node
                    m1=abs(graph[j][fixed-2]-graph[j][z-2])
                    m2=abs(graph[j][-1])-m1
                    temp_array.append(min(m1,m2))
                    continue
        else:
            temp_array=[0]
        beg.append(temp_array)
        temp_array2=[]
        if sizes[j]>=2:
            for z in range(1,sizes[j]+1):
                fixed=e
                if fixed==z:
                    temp_array2.append(0)
                    continue
                if z==1:
                    m1=graph[j][fixed-2]
                    m2=graph[j][-1]-m1
                    temp_array2.append(min(m1,m2))
                    continue
                else:
                    m1=abs(graph[j][fixed-2]-graph[j][z-2])
                    m2=abs(graph[j][-1])-m1
                    temp_array2.append(min(m1,m2))
                    continue
        else:
            temp_array2=[0]
        end.append(temp_array2)
                           
        distances=[]
        for j in range(n):
            distances.append(beg[j][ new_junction[j][1]-1])#we have to compute distance wrt to some junction that contains starting node of the next cycle which is new_junction[j+1][0] above
        new_distances=[distances[0]]
        for j in range(1,n):
            new_distances.append(distances[j]+new_distances[j-1])#prefix sum of distances

        for Q in range(q):
            if query[Q][1]>query[Q][3]:
                c1,c2=query[Q][3],query[Q][1]#c2 take larger cycle number first
                v1,v2=query[Q][2],query[Q][0]
            else:
                c1,c2=query[Q][1],query[Q][3]
                v1,v2=query[Q][0],query[Q][2]
            if c1==1:
                x1=connections[c2-2]#if we choose first cycle
                x2=connections[-1]-x1
            else:
                x1=abs(connections[c2-2]-connections[c1-2])#either come from first bridge or second bridge
                x2=abs(connections[-1]-x1)
            x1+=end[c1-1][v1-1]#either choose clockwise(x1) or anticlockwise cycles(x2)
            x1+=beg[c2-1][v2-1]
            x2+=beg[c1-1][v1-1]
            x2+=end[c2-1][v2-1]
            if n>2:
                if c2-c1>1:#this also can be done in 2 more ways to handle middle circle as well
                    foo=(new_distances[c2-1-1]-new_distances[c1-1])#since distances is handled wrt (beg)inning this will give us the prefix sum of inter-cyclic distances(using beg and new_junction)
                    x1+=foo
                    bar=new_distances[-1]-(foo+distances[c1-1]+distances[c2-1])#since new_distances is a sum so we have to subtract distances of c1,c2 and foo
                    x2+=bar#bar comes from total prefix sum of distances -(above mentioned way foo) -(c1,c2)
                if c2-c1==1:#then foo=0
                    bar=new_distances[-1]-(distances[c1-1]+distances[c2-1])
                    x2+=bar#so middle circle distances+gaps contribute to anticlockwise x2
                print(min(x1,x2))
            else:
                print(min(x1,x2))
                        
                           
                
                           
                        
#Hamming Distance
t = int(input())
 
for x in range(t):
    n = int(input())
    arrA = [int(i) for i in input().split()]
    nc=0
    mydb={}
    for i in range(n):
        #try:
        if arrA[i] in mydb:
           mydb[arrA[i]][1] +=1
           mydb[arrA[i]][2] = i
           nc+=1
        else:
        #except:
            mydb[arrA[i]] = [i,1,-1]
    #singles = [0] * n
    ci=0
    si = nc
    ns = n - (2*nc)
    barray = [0] * (n+2)
    index = [-1]*n
    adj = 1
    if nc==1:
        adj=2
    else:
        adj=1
    for num in arrA:
        tmplist = mydb[num]
        if tmplist[1]==1:
            barray[si+adj] = num
            index[si] = tmplist[0]
            si+=1
        elif tmplist[1]==2:
            barray[ci+adj] = num
            index[ci] = tmplist[0]
            barray[ci+nc+ns+adj] = num
            index[ci+nc+ns] = tmplist[2]
            ci+=1
            tmplist[1] = 0
    if adj==1:
        barray[0] = barray[n]
    elif adj==2:
        barray[0] = barray[n]
        barray[1] = barray[n+1]
    #print(barray) 
    newB = [0]*(n)
    for i in range(n):
        newB[index[i]] = barray[i]
    if n==3 and nc==1:
        print(2)
    elif n==1 or (n==2 and nc==1):
        print(0)
    else:
        print(n)
    for i in range(n):
        print(newB[i],end=' ')
    print() 
        
#Eugene and Big Number
def power(powersOf10,exponent,M):
    result=1
    for i in range(0,64):
        if ((exponent>>i)&1!=0):
            result=(powersOf10[i]*result)%M
    return result

T=int(input())
for _ in range(T):
    A,N,M=map(int,input().split())#A<=10**9,N<=10**12,M<=10**9
    result=[0]*64#So we want to find (str(A)*N)%M=(A*10^(len(A))+A)%M
    powersOf10=[0]*64#So we have to focus on finding 10^len(A)%M
    powersOf10[0]=10
    for i in range(1,len(powersOf10)):
        powersOf10[i]=(powersOf10[i-1]*powersOf10[i-1])%M#We will precompute all the powersOf10
    result[0]=A%M#Such that powers are sufficient to represent a 64-bit binary representation of len(A)
    powers=[0]*(64)
    powers[0]=power(powersOf10,len(str(A)),M)
    for i in range(1,64):
        result[i]=((result[i-1]*powers[i-1])%M+result[i-1])%M
        powers[i]=(powers[i-1]*powers[i-1])%M#(10^(len(A))) will become 10^(2*(len(A))) in next step
    ans=0
    for i in range(0,64):
        if ((N>>i)&1!=0):
            ans=((ans*powers[i]%M)+result[i])%M#Use dot product to get the ans as we are going to overshoot length>N*(len(A)) because we are using exponential rate
    print(ans)
    
#810D
def get(l,r):
    print(1,l,r)
    return input()=='TAK'

def f(l,r):
    while l<r:
        mid=(l+r)>>1
        if get(mid,mid+1):r=mid
        else:l=mid+1
    return l

n,k=map(int,input().split())
x=f(1,n)
y=f(1,x-1)
if x==y or not get(y,x):
    y=f(x+1,n)
print(2,x,y)
exit()

#414A
n,k=map(int,input().split())
if k<n//2 or (k!=0 and n==1):
    exit(print(-1))
ans=[0]*n
ans[0]=k-n//2+1
if n>1:
    ans[1]=2*ans[0]
    for i in range(2,n):
        ans[i]=ans[i-1]+1
print(*ans)


#Boredom
n=int(input())
cnt,f=[0]*(1000001),[0]*(n+1)
l=list(map(int,input().split()))
for i in l:
    cnt[i]+=1
f[1]=cnt[1]
for i in range(2,1+n):
    f[i]=max(f[i-1],f[i-2]+cnt[i]*i)
print(f[n])

        

#439A
n,d=map(int,input().split())
t=list(map(int,input().split()))
x=sum(t)
if (d-x)//10>=(n-1):#t1+10+t2+10+t3=(n-1)*10+sum(t)<=d
    if d>=x:#if d<x not all songs will be sung
        print((d-x)//5)#either 1 at beginning and rest 2 jokes will be cracked till the end
    else:print(-1)
else:print(-1)
    
#476A
n,m=map(int,input().split())
if n<m:exit(print(-1))
t=(n+1)//2
s=(t-1)//m*m+m
print(s)
        
        

#464A ord('a')=97
n, p = map(int, input().split())
t = [ord(c) - 97 for c in input()] + [27, 27]
#t=[input(),27,27] to consider the case when n=1 and n=2
for k in range(n - 1, -1, -1):
    for i in range(t[k] + 1, p):
        if i - t[k - 1] and i - t[k - 2]:#check substring of length 2 and length 3 we cannot replace b in cdb
            a, b = min(t[k - 1], 2), min(i, 2)#as we want lexicographiclaly next one
            if a == b: a = 1#Example 3 4 cdb=>dab
            t = t[: k] + [i] + [3 - a - b, a, b] * (n // 3 + 1)
            print(''.join(chr(i + 97) for i in t)[: n])
            exit(0)
print('NO')

    
#499A
n,x=map(int,input().split())
curr,total2,total=1,0,0
for i in range(n):
    l,r=map(int,input().split())
    if l>(curr+x):
        total2=l-(curr+x)
    elif(l==curr+x):
        total2=0
    total+=(r-l+1)+total2
    curr=r+1
print(total)

        

#486B
n,m=map(int,input().split())
a=[[int(x) for x in input().split()]for j in range(m)]
b=[[int(1) ]*n for j in range(m)]
for i in range(n):
    for j in range(m):
        if a[i][j]==0:#all the row and column elements are necessarily 0
            for k in range(n):
                b[i][k]=0
            for t in range(m):
                b[t][j]=0
c=[[int(0) ]*n for j in range(m)]
for i in range(n):
    for j in range(m):
        x=c[i][j]
        for k in range(n):
            x=x or(b[i][k])
            if x==1:
                break
        for t in range(m):
            x=x or (c[j][t])
            if x==1:break
        c[i][j]=x
flag=0
for i in range(n):
    for j in range(m):
        
        if a[i][j]!=c[i][j]:
            flag=1
            break
if(flag==0):
    print('YES')
    for i in range(n):
        print(*a[i])
else:
    print('NO')
#500A
n,t=map(int,input().split())
cell=[0]*(2+n)
x=list(map(int,input().split()))
for i in range(1,n):
    cell[i]=x[i-1]


visited=[False]*(2+n)
currIndex=1
while (currIndex!=t and visited[currIndex]==False) :
    nextIndex=(currIndex)+cell[currIndex]
    visited[currIndex]=True
    currIndex=nextIndex
if(currIndex==t):print('YES')
else:print('NO')
#binary search of i*j
m,n,k=map(int,input().split())
low,r=1,250000000001
while(low<r):
    mid=(low+r)//2
    q=0
    for i in range(1,1+n):
        q+=min(mid//i,m)#i*j=mid=>j=mid//i
    if(q<k):#q is compared wih k where 1*1<=k<=n*m in worst case q will go n*m then r will become mid mid will shift such that it is found in table ,finally at low>=r low=kth largest element
        low=mid+1
    else:
        r=mid
print(low)

#155B
total=int(input())

points,bi,counter=[],[],1
for i in range(total):
    l=list(map(int,input().split()))
    a,b=l[0],l[1]
    counter+=b
    if b:
        points.append(a)#points to be added corresponding to max b
    else:
        bi.append(a)#points to be added according to max b
points.sort(reverse=True)
bi.sort(reverse=True)
alen,blen=len(points),len(bi)
minim=min(total,counter)
ans=0
if minim>alen:
    extra=minim-alen
    ans+=sum(points)
    ans+=sum(bi[:extra])
else:
    ans+=sum(points[:minim])
print(ans)



#514A
s=input()
if  s[0]=='9' or s[0]=='1' or s[0]=='2'  or s[0]=='3' or s[0]=='4':print(int(s[0]),end='')
else:print(9-int(s[0]),end='')
for i in range(1,len(s)):
    if s[i]<='4'  :print(int(s[i]),end='')
    else:print(9-int(s[i]),end='')
#550C(digits outputted can be any number divisible by 8)

s=(input())
ans='NO'
for i in [str(i) for i in range(1000) if i%8==0]:
    t=-1
    for c in i:#i=str(0),str(8),str(16)=>c=1,6 it will try to look for 6 after 1
        t=s.find(c,t+1)
        if t==-1:
            break
    if t!=-1:ans='YES\n'+i#will print the last i divisible by 8
print(ans)
#OR

def solve():
    n="00"+input()
    for j in (int(n[i]+n[j]+n[k]) for i,j,k in combinations(range(len(s)),3)):
        if j%8==0:
            print('YES\n')
            print(j)
            exit()
    print('NO')

    

#732B
n,k=map(int,input().split())
walks=list(map(int,input().split()))
Owalks=[0]*n
for i in range(n):
    Owalks[i]=walks[i]
ans=0
for i in range(1,n-1):
    if walks[i]+walks[i-1]<k:
        walks[i]+=(k-(walks[i]+walks[i-1]))
if(n>1):
    while walks[n-2]+walks[n-1]<k:
        walks[n-1]+=1

for i in range(1,n):
    ans+=walks[i]-Owalks[i]
print(ans)
print(*walks)
#CodeChef(GoodSubstrings)
t=int(input())
for _ in range(t):
    s=input()
    ans=0
    if(len(s)>1):
        def calc(s):
            l1,s1,i=[],'',0
            n=len(s)
            while i<n:
                if i+1<n and s[i]==s[i+1]:#Lets say ababcba if string consists of some equal say ddddd=>ans=1+2+3+4=4(4+1)/2=(5-1)5/2
                    s1+=s[i]
                else:
                    s1+=s[i]
                    l1.append(s1)
                    s1=''
                i+=1
            ans=0
            print(l1)
            if(len(l1)==1):#All of them are equal say rrrrrrr
                if(len(l1[0])>=2):
                    ans=(len(l1[0]))*((len(l1[0]))-1)//2
            else:
                if(len(l1)==2):
                    ans=(len(l1[0]))*((len(l1[0]))-1)//2
                    ans+=(len(l1[1]))*((len(l1[1]))-1)//2
                else:
                    n1=len(l1)
                    for i in range(0,n1-2):#lets say aa,bbbbbb,aa
                        if l1[i][0]==l1[i+2][0]:
                            ans+=1#then 1 is added corresponding to last a of first aa anf first a of last aa
                    for i in range(n1):#lets say bbbbbb
                        if(len(l1[i]))>=2:
                            ans+=(len(l1[i]))*((len(l1[i]))-1)//2
            return ans
                            
        ans=calc(s)
    print(ans)
    
            
#221A
n=int(input())
p=[]
for i in range(n):
    p.append(i+1)
print(p[-1],end=' ')
for i in range(0,n-1):
    print(p[i],end=' ')
  
def sortAccordingToNeighbours(firstNeighbour,deletedString, secondNeighbour):
    l=len(firstNeighbour)
    newarr=[]
    x,y,z=firstNeighbour[0]+secondNeighbour[0],firstNeighbour[0],secondNeighbour[0]
    it=0
    for i in range(1,len(l)):
        if x >firstNeighbour[i]+secondNeighbour[i]:
            x=firstNeighbour[i]+secondNeighbour[i]
            it=i
        elif x==firstNeighbour[i]+secondNeighbour[i]:
            if y>firstNeighbour[i]:
                y=firstNeighbour[i]
                it=i            
            elif z>secondNeighbour[i]:
                z=secondNeighbour[i]
                it=i
    return deletedString[it]
            
s=list(input())
maxs=5000
n=len(s)
k=0
N=n
while(N>0):
    k+=1
    N=N//2
if(n%2==0):k-=1
m,firstNeighbour, deletedString, secondNeighbour={} ,[],[],[]
for i in range(k):#length 2^(i-1)
    x=pow(2,i)
    for j in range(0,n-(x+1)):
        firstNeighbour.append(s[j])
        deletedString.append(s[j+1:j+x+1])
        secondNeighbour.append(s[j+x+1])
    UpdatedString=sortAccordingToNeighbours(firstNeighbour,deletedString, secondNeighbour)
    
    

        
#938B
n=int(input())
arr=list(map(int,input().split()))
myPosition=1
FriendPosition=10**6
time=0
maxT=0
for i in range(n):
    time=min(abs(myPosition-arr[i]),abs(FriendPosition-arr[i]))
    maxT=max(maxT,time)
print(maxT)

w='aeiouy'
n=int(input())
s='b'+input()
for i in range(1,n+1):
    if not(s[i] in w and s[i-1] in w):
        print(s[i],end='')

        

    

#n=p1^x1*p2^x2....
def isPrime(i):
    for k in range(2,int(i**0.5)+1):
        if i%k==0:
            return False
    return True
N=int(input())
count=0
if(isPrime(N)):print(1);quit()
for p in range(2,1+int(N**0.5)):
    while(N%p==0):
        count+=1
        N=N//p
print(count)

#282A
n=int(input())
plu=0
for i in range(n):
    op=input()
    if(op=='X++' or op=='++X'):
        plu+=1
    elif(op=='X--' or op=='--X'):plu-=1
print(plu)

#585E

n=int(input())
price=list(map(int,input().split()))
#735D
n=int(input())
if(n%2==1):#take 21,27 if number is odd,we will divide into prime
    if(isPrime(n)):
        print(1)
    else:
        print(3-isPrime(n-2))#3-isPrime(19)=2,3-isPrime(25)=3
else:
    print(1+(n>2))#we can always divide it in two primes for n>2 if n is even
#623B
n,a,b=map(int,input().split())
arr=list(map(int,input().split()))

#603B

p,k=map(int,input().split())
mod=10**9+7
if k==0:
    print(pow(p,p-1,mod))#f(0)=0,rest of the function can be anything
elif k==1:
    print(pow(p,p,mod))
else:
    ordeal=1
    curr=k
    while(curr!=1):#until k^m is congruent to 1(modp)
        curr=(curr*k)
        curr%=p
        ordeal+=1#choose numbers k,k*n,k^2*n,...k^(m-1)*n for a fixed f(n)
    print(pow(p,(p-1)//ordeal,mod))#except f(0)
#576A
n=int(input())
questions=[]
def prime(n):
    for i in range(2,1+int(n**0.5)):
        if n%i==0:
            return 0
    return 1
for i in range(2,n+1):
    if prime(i)==1:
        questions.append(i)
        k=2
        while i**k<=n:#p^2,p*3... because p[i]*p[j] will answer the question so we need to worry only p[i]*[i],p[j]*p[j] 
            questions.append(i**k)
            k+=1
print(len(questions))
print(*questions)
#817A
x1,y1,x2,y2=map(int,input().split())
x,y=map(int,input().split())#parity of moves should be same
print('Yes' if abs(x1-x2)%x==0 and abs(y1-y2)%y==0 and abs(x1-x2)/e%2==0 and abs(y1-y2)/e%2==0 else 'No')
#858A
from math import gcd
n,k=map(int,input().split())
print(10**k*n//gcd(10**k,n))
#840A
m=int(input())#d1=a1;d1+d2=a2;....d1+d2+d3+...dk+1=N+1
a=list(map(int,input().split()))#E[sigma di]=N+1 ; E[d1]+E[d2]+..E[dk+1]=(k+1)*E[d1]=>E[d1]=N+1/k+1
b=list(zip(map(int,input().split()),range(m)))#such that A[i]>=B[j] so an additional of index is attached 
a.sort(reverse=True)# F=sigma (A[i]+1)/(B[i]+1) to be maximum A[i] is maximum paired with B[i] minimal   
b.sort()
c=list(zip((t[1] for t in b),a))
c.sort()
print(*(t[1] for t in c))#Finally so pointer means to represent every value in this format a b c d ...

#838D
n,m=map(int,input().split())
n+=1#to make seats circular#2n^m is the ways of assignment without restriction
mod=10**9+7#1/n for each seat having a chance being empty,out of which(n-m) remains empty at the end
res=(pow(2*n,m-1,mod))*2*(n-m)#ans is no of ways nth seat is empty=((2n)^(m))*2*(n-m)//(2n)
print(res%mod)
#678C

n,a,b,p,q=map(int,input().split())
s=a*b//gcd(a,b)
ans=(n//a)*p+(n//b)*q-(n//s)*min(p,q)
print(ans)
        
#573A
_=input()
for t in map(int,input().split()):
    while t%2==0:
        t=t//2
    while t%3==0:
         t=t//3
    if not a:
        a=t
    elif a!=t:print("No")
print("Yes")

#665F
def PrimePiSegmentedSieveMeisselLehmer(n):
    
    ret=0
    m=1
    while(m*m<=n):
        high[m]=((n//m)-1)
        m+=1
    Mcpy=m
    
    for i in range(1,1+Mcpy):
        low[i]=i-1
        
    for p in range(2,Mcpy+1):
        if(low[p]==low[p-1]):continue
        s=min(n//(p*p),Mcpy-1)
        
        for x in range(1,s+1):
            if(x*p<=Mcpy-1):
                high[x]-=high[x*p]-low[p-1]
            else:
                high[x]-=low[n//(x*p)]-low[p-1]
                
        y=Mcpy
        while(y>=p*p):
            low[y]-=low[y//p]-low[p-1]
            y-=1
        
    for p1 in range(2,Mcpy):
        if low[p1]==low[p1-1]:
            continue
        ret+=high[p1]-low[p1]
    return ret

n=int(input())
high=[0]*(340000)
low=[0]*(340000)
print( PrimePiSegmentedSieveMeisselLehmer(n)+low[int(n**(1.0/3))])

n=int(input())
countABC=0
size=limit
for i in range(size):
    a=primes[i]
    if(a*a*a>n):break
    for j in range(i+1,size):
        b=primes[j]
        maxC=n//(a*b)
        if maxC<=b:break
        high=pi(maxC)#p3=count(n/a/b)-count(b)
        low=j+1
        countABC+=high-low

countA3B=0
for a in primes:
    maxB=n//(a*a*a)
    if(maxB<=1):break
    numB=pi(maxB)
    if(maxB>=a):numB-=1
    countA3B+=numB

countA7=0
for a in primes:
    if(a*a*a*a*a*a*a>n):break
    countA7+=1
print(countABC+countA3B+countA7)
        
        
#691F
n=int(input())
x=[0]*(3000001)
w=[0]*(3000001)

for ball in map(int,input().split()):
    x[ball]+=1
for i in range(1,3000001):#run a sieve
    for j in range(i,3000001,i):
        k=j//i
        if(k==i ):
            w[j]+=x[i]*(x[i]-1)#we don't want any pair of balls with equal index i.e C(x[i],1)*C(x[i]-1,1) when k=i
        else:
            w[j]+=x[i]*x[k]
for i in range(1,3000001):
    w[i]+=w[i-1]#after each iteration w[kvalue-1]=w[kvalue-1]+w[kvalue-2]+w[kvalue-3]+..w[1]+w[0]
m=int(input())
for kvalue in map(int,input().split()):
    print(n*(n-1)-w[kvalue-1])#we are subtracting the pairs with multiplication value less than kvalue
      
#742B
_,n=map(int,input().split())
m={}
ans=0
for x in map(int,input().split()):#a[i]^a[j]=x means a[i]^x=a[j](a[i]^(a[i]^a[j]))
    ans+=m.get(x^n,0)
    m[x]=m.get(x,0)+1#efficiently using dictionary 

print(ans)
#749A
N=int(input())
if N%2==0:
    print(N//2)
    for i in range(N//2):
        print(2,end=' ')
else:
    print(N//2)
    for i in range(N//2-1):
        print(2,end=' ')
    print(3)
#396B

def prime(n):#function for returning list containing all primes not exceeding n
    m=int(n**0.5)+1
    t=[1]*(1+n)
    for i in range(3,m):
        if t[i]:
            t[i*i::2*i]=[0]*((n-i*i)//(2*i)+1)
    return [2]+[i for i in range(3,n+1,2) if t[i]]

def gcd(a,b):
    c=a%b
    return gcd(b,c) if c else b

p=prime(31650)#print(10**4.5)~31622
def g(n):
    m=int(n**0.5)
    for j in p:
        if n%j==0:
            return True
        if j>m:
            return False
def f(n):
    a,b=n,n+1
    while g(a):#example 1/2.3+1/3.5=1/2-1/3+(1/3-1/5)-1/2(1/3-1/5)
        a-=1
    while g(b):#Inclusion-Exclusion principle
        b+=1#p=ab-2a+2n-2b+2
    p,q=(b-2)*a+2*(n-b+1),2*a*b
    d=gcd(p,q)
    print(str(p//d)+'/' +str(q//d))
for i in range(int(input())):
    f(int(input()))
#375D
cnt=[0]*10
for i in (1,6,8,9):
    cnt[i]=-1
n=input()
for i in n:
    cnt[int(i)]+=1
mod=[1869, 1968, 9816, 6198, 1698, 1986, 1896, 1869]
modCnt=0
for i in range(1,10):
    for j in range(cnt[i]):
        modCnt=(3*modCnt+i)%7#10 is congruent to 3(mod7)
    print(str(i)*cnt[i],end='')#non-zero digits
modCnt=(10000*modCnt)%7#multiply 10^4 by modCnt and then between 0 to 6
print(str(mod[7-modCnt])+'0'*cnt[0])#permutation of mod+zero digits

        
#303D
def ksm(x,y,mod):#x is base and y is a number
    if not y:#y=0
        return 1
    t=int(ksm(x,y>>1,mod))#1->10->11->100 or 1->2->3->4...
    if y&1:#y is non-zero
        return (t*t%mod )*( x%mod)#will give us the rotating number
    return t*t%mod#x=1

n,m=map(int,input().split())
for i in range(2,1+int((n+1)**0.5)):
    if (n+1)%i==0:
        print(-1)
        exit()
for i in range(m-1,1,-1):
    for j in range(2,1+int(n**0.5)):
        if n%j==0 and ksm(i,n//j,n+1)==1:#as y=0 no solution ,Also we are considering n//j if n is divisible by j because rotating number by multiplying it with smaler digits in base has larger probability of giving us the answer
            continue
        print(i)
        exit()
print(-1)

    
#236B
a,b,c=map(int,input().split())

d=2**30
p=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
t=[{} for i in range(101)]
ans={}
for i in p:
    j=i#consider i=2
    marked=1
    while j<101:
        for k in range(j,101,j):
            t[k][i]=marked#t[2][2],t[4][2]....t[100][2],t[4][2].....t[64][2],t[8][2],.....t[64][2]
        j=j*i#j=2*2,4*2,8*2,...
        marked+=1
s=0
for i in range(1,a+1):
    for j in range(1,b+1):
        q={}#example d(1.1.2) so first we calculate d(1.1) then using it we do d(ij*2)
        for x in t[i].keys()|t[j].keys():#here | is bitwise OR operator meaning t[i] and t[j] dictionaries having maximum number of keys
            q[x]=t[i].get(x,0)+t[j].get(x,0)#storing sum of each number's d value using get(key,defaultvalue)  returns default value if key not found
        ij=i*j
        for k in range(1,c+1):
            ijk=ij*k
            if ijk in ans:
                s+=ans[ijk]
            else:
                y=1
                for x in t[k].keys()|q.keys():
                    y=y*(q.get(x,0)+t[k].get(x,0)+1)#add 1 for the number itself
                ans[ijk]=y
                s+=y
print(s)
#850B
n=int(input())
arr=list(map(int,input().split()))
m=input()
ans,x=0,0
for i in range(n):
    if m[i]=='1':
        ans=max(ans+arr[i],x)
    x+=arr[i]
print(ans)

#776B
def isPrime(i):
    for k in range(2,int(i**0.5)+1):
        if i%k==0:
            return False
    return True

n=int(input())
if n==1 or n==2:
    print(1)
    exit()
else:
    print(2)
    for i in range(2,1+1+n):
        if isPrime(i):
            print(1,end=' ')
        else:
            print(2,end=' ')


#630A
n,m=map(int,input().split())
print(sum( (m+(i%5))//5 for i in range(1,n+1) ))#counting the numbers in m with remainder 0 to 4=(m+rem)/5

#575H
def f(n, mod=10**9+7):
    ans = 1
    for i in range(1, n + 1): ans = ans * i % mod
    return ans
    
def g(n, mod=10**9+7):
    num1 = f(n * 2)#C(2N,N)
    den1 = f(n) ** 2 % mod
    return num1 * pow(den1, mod - 2, mod) % mod#using the inverse function 
    
n = int(input()) + 1
print(g(n) - 1)
#577A
n,x=map(int,input().split())
print(sum(x%i==0 and x//i<=n for i in range(1,n+1)))

#568A
p,q=map(int,input().split())
n=1200000
t=[0,q]*600000
for i in range(3,1096,2):
    if t[i]:
        for j in range(i*i,n,i):#another way of discarding primes
            t[j]=0
t[1],t[2]=-p,q-p
for i in range(3,10):
    t[i]-=p
for i in range(1,1000):
    u=str(i)
    v=u[::-1]
    for j in '0123456789':
        k=int(u+j+v)#a palindrome is formed of form str(u)+str(j) +str(v)
        if k<n:
            t[k]-=p
    t[int(u+v)]-=p#a palindrome is formed of form str(u)+str(v)
j=s=0

for i,q in enumerate(t):
    s+=q
    if s<=0:#aq-bp<=0 means aq<=bp
        j=i
print(j)

#584D


n=int(input())
if isPrime(n):
    print(1)
    print(n)
    quit()
i=n
while not isPrime(i):
    i-=2
p1000=[i for i in range(2,3000) if isPrime(i)]
rem=n-i
if rem==2:
    print(2)
    print(2,i)
    quit()
print(3)
for j in p1000:
    if rem-j in p1000:
        print(i,j,rem-j)
        quit()
#615D
n = int(input())
exps = {}
for x in [ int(x) for x in input().split() ]:
	exps[x] = exps.get(x,0)+1
r,m = 1,1
M = 1000000007
P = M-1#a^(mod-1)=1%mod=>a^x=(a^(x%(mod-1))%mod)
for p,e in exps.items():
	E = (e*(e+1)//2)%P #f(p^k)=p^(k*k+1/2)
	E = E*m%P#f(x)=x^(d(x)//2)
	r = pow(r,e+1,M)*pow(p,E,M)%M#f(ab)=(f(a)^d(b))*(f(b)^d(a))
	m = m*(e+1)%P
print(r)

#616E
mod=10**9+7
n,m=map(int,input().split())
ans=0#ans=n%1+n%2+...n%m=sigma(n%i)=sigma(n-(n//i)*i)=nm-sigma(n//i)*(i)
if m>n:
    ans+=(m-n)*n
    m=n
for j in range(2,3*(10**6)):#3*10**6 is close to sqrt(10**13)
    x=n//j+1
    if x>m:
        continue
    ans+=(n%x)*(m-x+1)-(m-x+1)*((m-x)//2)*(j-1)
    m=x-1
    ans=ans%mod
for j in range(1,m+1):
    ans+=(n%j)
print(ans%mod)
#678D
A,B,n,x=map(int,input().split())
mod=10**9+7#g1(x)=f(x)=Ax+B,g2(x)=f(Ax+B)=A(Ax+B)+B=A^2x+A*B+B
#g3(x)=f(A^2x+A*B+B)=A(A^2x+A*B+B)+B=A^3x+A^2*B+A*B+B
#gn(x)=A^nx+A^(n-1)*B+A^(n-2)*B...+A*B+B
val=0
for i in range(n):
    val+=((((A**(i))%mod)*(B%mod))%mod)
val+=((((A**(n))%mod)*(x%mod))%mod)    

print(val%mod)
#757B
n=int(input())
pokemon=list(map(int,input().split()))
k=1
a=[0]*100001
for i in (pokemon):
    a[i]+=1
for i in range(2,100001):
    p=0
    for j in range(i,100001,i):#what we are doing is finding max of common prime factor for each pokemon
        p+=a[j]
    k=max(k,p)
print(k)
    
#762A
n,k=map(int,input().split())
divisors=[]
for i in range(1,int(n**0.5)+1):
    if n%i==0:
        divisors.append(i)
        divisors.append(n//i)

if len(divisors)<k:
    print('Impossible')
else:
    divisors=list(set(sorted(divisors)))
    print(divisors)
    print(divisors[k-1])
#922A
x,y=map(int,input().split())
print(['No','Yes'][(x,y)==(0,1) or x+2>y>1 and (x-y)%2])
print(chr(27))
#33B
def build_graph(a):
    c=float('inf')
    w=[[c for i in range(26)]for j in range(26) ]
    for i in range(26):
        w[i][i]=0
    for b in a:
        if w[ord(b[0])-97][ord(b[1])-97]>b[2]:#so ord['a']=97
            w[ord(b[0])-97][ord(b[1])-97]=b[2]#so what is does if multiple inputs of same type say b[0]='c' and b[1]='f' are present it takes the one with the least cost
            
    for k in range(26):
        for i in range(26):
            for j in range(26):
                if w[i][j]>w[i][k]+w[k][j]:#it may be the case that indirect swaps may be less costly than directly swapping say 'a','x' has more cost than 'a','y'+'y','x'
                    w[i][j]=w[i][k]+w[k][j]
    return w
        
def transfer(s,t,a):
    if len(s)!=len(t):return -1
    r=''
    z=0
    w=build_graph(a)
    for d,p in zip (s,t):
        if d==p:
            r+=d
        else:
            c=float('inf')
            q=''
            i=ord(d)-97
            j=ord(p)-97
            for k in range(26):
                v=w[i][k]+w[j][k]
                if c>v:
                    c=v
                    q=chr(k+97)#chr is used ascii value back into character chr(65)='A'
            if c==float('inf'):
                return -1
            z+=c
            r+=q
    r=str(z)+'\n'+r
    return r
s=input()
t=input()
n=int(input())
a=[]
for c in range (n):
    x,y,z=map(str,input().split())
    a.append([x,y,int(z)])
print(transfer(s,t,a))

#80A
def isPrime(x):
    for i in range(2,int(x**0.5)+1):
        if x%i==0:
            return 0
    return 1
n,m=map(int,input().split())
flag=1
for i in range(n+1,m):
    if isPrime(i):
        flag=0
        break
flag1=0
if isPrime(m):
    flag1=1
print('YES' if(flag and flag1) else 'NO')
    
    
#69C(very important applications of dictionary)
k,n,m,q=map(int,input().split())
o=[input() for i in range(n)]
c={}
for _ in range(m):
    s=input()
    a,b=s.split(': ')
    e=[]
    for x in b.split(', '):
        u,v=x.split()
        e+=[(u,int(v))]
    c[a]=e
g=[{} for _ in range(k+1)]
for _ in range( q):
    a,b=input().split()
    a=int(a)
    v=g[a]
    v[b]=v.get(b,0)+1
    for x in c:
        f=1
        for y in c[x]:
            if v.get(y[0],0)<y[1]:
                f=0
                break
        if f:
            for y in c[x]:
                v[y[0]]-=y[1]
            v[x]=v.get(x,0)+1
            break
for i in range(1,k+1):
    v=sorted(x for x in g[i] if g[i][x])
    print(len(v))
    for x in v:
        print(x,g[i][x])
#90A
r,g,b=map(int,input().split())
x=(r+1)//2-1#maximum only 2 person can sit
y=(g+1)//2-1
z=(b+1)//2-1
x1=3*x#number of cable cars is divisible by 3 and periodicity of 1 minute
x2=3*y+1
x3=3*z+2
maxx=max(x1,x2)
print(max(maxx,x3)+30)
#60B
n,m=map(int,input().split())
s=[[] for i in range(n)]
for i in range(n):
        s[i]=input()
c=0
for i in range(n):
    for j in range(m):
        k,l=0,0
        while(k<m or l<n):
            if s[i][j]==s[i][k] and k<m:
                c+=1
            if s[i][j]==s[l][j] and l<n:
                c+=1
            if(k<m):k+=1
            if(l<n):l+=1
        if(c==2):
            print(s[i][j],end='')
        c=0
            

    
#88A
n=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B', 'H']
#pos1=arr2[i],pos2=arr2[(i+1)%3],pos3=arr2[(i+2)%3]
inputChords=sorted(n.index(x) for x in input().split())
c=(inputChords[1]-inputChords[0],inputChords[2]-inputChords[1])
if c in ((4, 3), (3, 5), (5, 4)):
    print('major')
elif c in ((3, 4), (4, 5), (5, 3)):#then either (pos1+4)%12==pos2 and(pos2+3)%12==pos3 or(pos1+4)%12==pos3 and(pos3+3)%12==pos2
    print('minor')
else:
    print('strange') 
#83A
n=int(input())

curr=next=ans=0
for i in input().split():
    curr=(next==i)*curr+1#here a[i-1] is compared with a[i] just like nC2
    next=i#this next is a[i-1] in i
    ans+=curr
print(ans)
#82A
l=['Sheldon','Leonard','Penny','Rajesh','Howard']
n=int(input())
while n>4:
    n=(n-5)//2#what we have done multiplied each member by 2 so to get ans we backtrack
print(l[n])
#83E-5
a,b,k=map(int,input().split())
def rho(n,primes,s=0):#gives number of integers<=n which are divisible by no primes<k,s=0 is just an starting index
    if s>=len(primes) or n<primes[s]:
        return 0
    ret=0
    for i in range(s,len(primes)):
        next=n//primes[i]#ret=n/primes[0]-n/primes[0]*(primes[1])+n/primes[0]*(primes[1])*(primes[2])....
        if next==0:
            break
        ret+=next
        ret-=rho(next,primes,i+1)
    return ret
def answer(c):
    if c<k:
        return 0
    if k*k>c:
        return 1
    primes=[2] if k>2 else []
    for d in range(3,k,2):#generate primes less than k 
        for p in primes:
            if d%p==0:
                break
        else:primes.append(d)
    return c//k-rho(c//k,primes)#to count the number that are not divisible by any number between 2 and k-1 yet divisible by k
def isPrime(a):
    for d in range(2,int(a**0.5)+1):
        if a%d==0:return True
    return False
if isPrime(k):
    print(0)
else:print(answer(b)-answer(a-1))
#76E
N=int(input())#sigma (i=1 to N) sigma (j=1 to i-1) (xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)
for _ in range(N):
    x,y=map(int,input().split())
    X+=x
    Y+=y
    A+=x*x+y*y
print(A*n-X*X-Y*Y)

#74A
n=int(input())
maxx=[]
for _ in range(n):
    handle,*abcde=input().split()
    plus,minus,*abcde=map(int,abcde)
    maxx.append((plus*100-minus*50+sum(abcde),handle))
print(max(maxx)[1])
#71B
n,k,t=map(input().split())
t=(n*k*t)//100
for i in range(n):
    print(min(max(0,t-i*k),k),end=' ')#at n=1 a[1]<=kt/100,at n=2 a[1]+a[2]+a[3]<=3kt/100=>a[3]<=3kt/100-a[1]-a[2]
#71A
n=int(input())
for i in range(n):
    word=input()
    x=len(word)
    if x>10:
        
        print(word[0],end='')
        print(x-2,end='')
        print(word[-1])
    else:
        print(word)
#72I
x=int(input())
d=set()
n=x
while(n>0):
    if(n%10):
        d.add(n%10)
    n=n//10
happyCount=0
for i in d:
    if x%i==0:
        happyCount+=1
if(happyCount==0):print('upset')
elif(happyCount==len(d)):print('happier')
else:print('happy')
#79B
n,m,k,t=map(int,input().split())
land=[]
crops=['Carrots','Kiwis','Grapes']       
for i in range(k):
    a,b=map(int,input().split())
    a-=1
    b-=1
    land.append(a*m+b)#am+b is the number of cells including waste ones that will be sweeped before (a,b)
land.append(n*m)   
land.sort()            
for i in range(t):
    a,b=map(int,input().split())
    a-=1
    b-=1
    c=a*m+b
    wasteCells=0
    while land[wasteCells]<c:
        wasteCells+=1
    print('Waste' if land[wasteCells]==c else crops[(c-wasteCells)%3])
#81C
n=int(input())
a,b=map(int,input().split())
if a==b:print('1 '*a+'2 '*b)
else:
    t=[[] for i in range(6)]
    for i,j in enumerate(map(int,input().split())):t[j].append(i)#here i is index number and j is input value,stores larger number with smaller index
    
    if b<a:
        t=t[1]+t[2]+t[3]+t[4]+t[5]#here add all the elements of different list from map to one list
        
        t.reverse()
        
        p=['1']*n
        for i in range(b):#to maximize total average score we have to assign it in such a way that larger numbers with smaller count to min(a,b)
            p[t[i]]='2'#numerator will compensate for denominator
        print(' '.join(p))
    else:
        t=t[5]+t[4]+t[3]+t[2]+t[1]
        
        p=['2']*n   
        for i in range(a):
            p[t[i]]='1'
        print(' '.join(p))
#81A
s=[]
for c in input():
    if s and c==s[-1]:
        s.pop()
    else:
        s+=[c]
print(''.join(s))


#69B
n,m=map(int,input().split())
a=[0]*110
b=[0]*110
for i in range(1,n+1):
    b[i]=10**21
for i in range(m):
    l,r,t,c=map(int,input().split())
    for j in range(l,r+1):#to find a winner at each and every section(that is the competitor with min time)
        if b[j]>t:
            b[j]=t
            a[j]=c
c=0
for i in range(1,n+1):
    c+=a[i]
print(c)
#75A
I=lambda x:int(str(x).replace('0',''))
a=int(input())
b=int(input())
print("YNEOS"[I(a)+I(b)!=I(a+b)::2])
#72H
u=(input())
v=int(u)
sign=0
special=[]
if v<0:
    sign=-1
    for i in range(1,len(u)):
        special.append(u[i])

spec=[]
i=0
for j in range(len(special)):
    if special[j]=='0':
        i+=1
    else:break
for j in range(i,len(special)):
    spec.append(special[j])


spec2=[]
i=len(spec)-1
for j in range(1,len(spec)):
    if spec[-j]=='0':
        i-=1
    else:break
for j in range(1+i):
    spec2.append(spec[j])
if(sign==-1):
    print('-',end='')
for i in range(1,1+len(spec2)):
    print(spec2[-i],end='')
#68B
n,k=map(int,input().split())
Energy=list(sorted(map(int,input().split())))
s=sum(Energy)
l,r=0,1000
while r-l>1e-1:
    m=(r+l)/2
    S=sum(i-m for i in Energy if i>m)
    if (s-k*S/100)>n*m:l=m#n*m is done to check if Energy is distributed in right or left 
    else: r=m
print(l)
#54C
N=int(input())
T=[1]+[0]*N
for i in range(N):
    L,R=map(int,input().split())
    c=1
    n=0
    while c<=10**18:#this loop will run 18 times as 1<=L,R<=10**18
        n+=max(0,min(R,2*c-1)-max(L,c)+1)#we have to develop all segment of random good numbers of the form[10^î,2*10^i-1] between L,R
        c=c*10
    p=float(n)/(R-L+1) 
    B=T
    T=[B[0]*(1-p)]+[0]*N#in the next iteration T[0]=1-p,then (1-p)^2 ,where p=n/R-L+1
    for j in range(1,N+1):
        T[j]=B[j]*(1-p)+B[j-1]*p#this is telling us how many j-1th particle are good among j number of particles
K=int(input())
print(sum(T[(N*K+99)//100:]))#probability that first digits of atleast K% of N values will be 1.Since L,R is of the form [1;1] [10;19] [100;199]
#63C
n=int(input())
data=map( lambda a:(a[0],int(a[1]),int(a[2])) ,[input().split() for i in range(n)]  )
ans,cnt='',0
def valid(s):#so basically c is common elements with misplaced position
    for a,b,c in data:#xis checking how many common elements does the 2 set have
        x,y=len(set(s) & set(a)),len([i for i,j in zip(a,s) if i==j])#y is checking how many elements are being matched
        if x-y!=c or y!=b:
            return 0
    return 1
for s in permutations('0123456789',4):
    if valid(''.join(s)):
        ans=s
        cnt+=1
print(''.join(ans) if cnt==1 else 'Need more data' if cnt>1 else 'Incorrect data')
#59B
n=int(input())
Petals=list(sorted(map(int,input().split())))
x=sum(Petals)
smallestOddNo=0
for i in range(n):
    if (Petals[i]%2):
        smallestOddNo=Petals[i]
        break
if (x%2):print(x)
elif((x-smallestOddNo)%2):print(x-smallestOddNo)
else:print(0)
#58B
n=int(input())
for i in range(n,0,-1):
    if not(n%i ):
        n=i
        print(int(n),end=' ')
#58A
hello=['h','e','l','l','o']
strlist=list(input())
flag=0
for i in range(len(strlist)):
    if strlist[i]=='h':
        for j in range(i):
            strlist[j]=0
        flag=1
        k=i+1
    elif strlist[i]=='e' and flag==1:
        for j in range(k,i):
            strlist[j]=0
        flag=2
        k=i+1
    elif strlist[i]=='l' and flag==2:
        for j in range(k,i):
            strlist[j]=0
        flag=3
        k=i+1
    elif strlist[i]=='l' and flag==3:
        for j in range(k,i):
            strlist[j]=0
        flag=4
        k=i+1
    elif strlist[i]=='o' and flag==4:
        for j in range(k,i):
            strlist[j]=0
        for j in range(i+1,len(strlist)):
            strlist[j]=0
newlist=[]
for i in range(len(strlist)):
    if strlist[i]!=0:
        newlist.append(strlist[i])
x=(list(newlist))
print('YES' if x[0]=='h' and x[1]=='e' and x[2]=='l' and x[3]=='l' and x[4]=='o' else 'NO')
#48C
n=int(input())
X=sorted(map(int,input().split()))
i=1-n#suppose we reverse the array or try to see ,it is (n-1)th pos from the end
j=n-2#18  18    51     80     86
s=0###    i   X[-3]    j     X[-1]
n=X[j]-X[i]#80-18=62
while s<n:#this is done when suppose when X[i]-X[j]=0 as we increase i and decrease j
    l,r=X[i]-X[0],X[-1]-X[j]#l=0,r=6
    if l<r:
        if n<=l:
            s=n
            break
        i+=1#this is done to try to increase left pointer and also to change value of n after this step
        s=l#s=0
    else:
        if n<=r:
            s=n
            break
        j-=1
        s=r
    n=X[j]-X[i]#80-51=29
s=s/2
print(s)
print(s+X[0],X[j]-s,X[-1]-s)
    

#55B
from itertools import permutations
nums=list(map(int,input().split()))
o=input().split()
def f(op,a,b):return a*b if op=="*" else a+b
res=1000**4
for a,b,c,d in permutations(nums):
    res=min(res,min(  f(o[2],f(o[1],f(o[0],a,b),c),d) ,f(o[2],f(o[1],c,d),f(o[0],a,b))  ))
#38C
n,l=map(int,input().split())
a=list(map(int,input().split()))
max=0
for i in range(l,100):#l is the minimal length acceptable
    sum=0
    for j in range(n):
        sum+=(a[j]//i)*i#to find maximal area by cutting the lengths array according to width l
    if(sum>max):max=sum
print(max)
#46A
n=int(input())
x=2
l=[0]*(n)
l[0]=2
for i in range(1,n):
    l[i]=(l[i-1]+x)%n
    x+=1
for i in range(n-1):
    print(l[i],end=' ')
#48A
s1=input()#F
s2=input()#M
s3=input()#S
if(s1[0]=='r' and s2[0]=='s' and s3[0]=='s'):print('F')
elif(s1[0]=='p' and s2[0]=='r' and s3[0]=='r'):print('F')
elif(s1[0]=='s' and s2[0]=='p' and s3[0]=='p'):print('F')
elif(s1[0]=='r' and s2[0]=='p' and s3[0]=='r'):print('M')
elif(s1[0]=='s' and s2[0]=='r' and s3[0]=='s'):print('M')
elif(s1[0]=='p' and s2[0]=='s' and s3[0]=='p'):print('M')
elif(s1[0]=='s' and s2[0]=='s' and s3[0]=='r'):print('S')
elif(s1[0]=='r' and s2[0]=='r' and s3[0]=='p'):print('S')
elif(s1[0]=='p' and s2[0]=='p' and s3[0]=='s'):print('S')
else:print('?')
#54A
n,k=map(int,input().split())
c=int(input())
a=0
sumx=0
for i in range(c):
    b=int(input())
    sumx+=((b-1)-a)/k#within range ((i-1)-j)/k
    a=b
sumx=(n-a)/k#for last value of b received via a we have the consider n as well
print(sumx+c)#on c holidays receives present
#59A
s=input()
print([s.lower(),s.upper()][sum(map(str.isupper,s))>len(s)/2])
#44C
n,m=map(int,input().split())
days=[0]*(1+n)
for i in range(m):
    j,k=map(int,input().split())
    if j!=k:
        for j1 in range(j,k+1):
            days[j1]+=1
    elif j==k:
        days[j]+=1
flag=1
t=0
for i in range(1,n+1):
    if days[i]!=1:
        flag=0
        t=i
        break
if(flag):
    print("OK")
else:
    print(t,days[t])
#46B(Smart Way of implementation)
s=['S', 'M', 'L', 'XL', 'XXL']
t=list(map(int,input().split()))
def g(si):
    if si<0 or si>=5 or not t[si]:
        return False
    else:
        print(s[si])
        t[si]=-1
        return True
def f(si):
    for j in range(5):
        if g(si+j) or g(si-j):
            return
for i in range(int(input())):
    f(s.index(input()))
    
    
#32B
a=input()
a=a.replace("--","2")
a=a.replace("-.","1")
a=a.replace(".","0")
print(a)
#3D
from heapq import heappop,heappush
brackets=list(input())
n=len(brackets)
ans=0
openingk=0
heap=[]
for i in range(n):
    if brackets[i]=='(':
        openingk+=1
    elif brackets[i]==')':
        openingk-=1
    else:#brackets[i]=='?'
        OpeningBracketCost,ClosingBracketCost=map(int,input().split())
        ans+=ClosingBracketCost
        heappush(heap,(OpeningBracketCost-ClosingBracketCost,i))
        brackets[i]=')'
        openingk-=1
    if k<0:
        if len(heap)==0:
            break
        NewOpeningBracketCost,NewClosingBracketCost=heappop(heap)
        ans+=NewOpeningBracketCost
        brackets[NewClosingBracketCost]='('
        k+=2
if k!=0:
    print(-1)
else:
    print(ans)
    print(''.join(s))
#57C
def pow1(a,b,m):
    if b==0:return 1
    if b%2==0:return (pow1(a,b//2,m)**2)
    return (pow1(a,b-1,m)*a)%m
def rev(a,m):#((2*N-1)C(N))mod(m) can be calculated by multiplication instead division 
    return pow1(a,m-2,m)
def fact(a,m):
    t=a
    for i in range(1,a):
        t=(t*i)%m
    return t
def main1(n):
    m=10**9+7
    if n==1:return 1
    return (2*(fact(2*n-1,m)*rev(fact(n,m)*fact(n-1,m),m))-n)%m
print(main1(int(input())))
#47C(Crosswords)
from itertools import permutations
v=[]#first take permutations as 3-horizontal letters and 3-vertical letters
#possible permutation:-BAA,NEWTON,YARD,BURN,AIRWAY,NOD 
#satisfying 4 conditions 1.len of dimensions 2.equality of word W in NEWTON and AIRWAY
#3.equality of 2 corner words on upper left 4.equality of 2 corner words on lower right corners
for p in permutations(input() for i in range(6)):
    if( len(p[1])!=len(p[0])+len(p[2])-1 or len(p[4])!=len(p[3])+len(p[5])-1 ):continue
    elif( p[0][0]!=p[3][0] or p[0][-1]!=p[4][0] ):continue
    elif( p[1][0]!=p[3][-1] or p[1][len(p[0])-1]!=p[4][len(p[3])-1] or p[1][-1]!=p[5][0] ):continue
    elif( p[2][0]!=p[4][-1] or p[2][-1]!=p[5][-1]  ):continue
    else:
        x='.'*(len(p[1])-len(p[0]))
        y='.'*(len(p[1])-len(p[2]))
        c=[]
        c.append(p[0]+x)
        for i in range(1,len(p[3])-1):
            c.append(p[3][i]+'.'*(len(p[0])-2)+p[4][i]+x)
        c.append(p[1])
        for i in range(1,len(p[5])-1):
            c.append(y+p[4][len(p[3])+i-1]+'.'*(len(p[2])-2)+p[5][i])
        c.append(y+p[2])
        v.append(c)
print('\n'.join(sorted(v)[0]) if v else 'Impossible')
#49A
ans=input()
it=len(ans)
it=it-1
while(True):
    if ans[it]=='?' or ans[it]==' ':
        it=it-1
    elif ans[it]!=' ':
        break
print("YES" if (ans[it]=='a' or ans[it]=='e' or ans[it]=='i' or ans[it]=='o' or ans[it]=='u' )else "NO")
#12A
n=3
mat=[[0]*n for j in range(n)]
for i in range(n):
    mat[i]=input()
flag=1
for i in range(n):
    for j in range(n):
        if mat[i][j]=='X' and mat[i][j]!=mat[2-i][2-j]:
            flag=0
            break
if(flag):print("YES")
else:print("NO")
#10D
n=int(input())
first=list(map(int,input().split()))
m=int(input())
second=list(map(int,input().split()))
if m>n:
    n,m=m,n
    first,second=second,first
dp,prev=[0]*m,[-1]*m#dp[j]: LCIS ending at second[j]
#prev[j]: index of the second-to-last number for the LCIS
for i in range(n):
    max_len,last_index=0,-1
    for j in range(m):
        #dp(i,j):length of LCIS that between f[0:i+1] and s[0:j+1] ends at s[j]
        if first[i]==second[j] and dp[j]<max_len+1:
            dp[j]=max_len+1
            prev[j]=last_index
        #else: dp(i,j)=dp(i-1,j)
        elif first[i]>second[j] and max_len<dp[j]:
            max_len=dp[j]
            last_index=j

max_value,index_max=0,-1

for index,value in enumerate(dp):
    if value>max_value:
        max_value=value
        index_max=index
print(max_value)
if max_value>0:
    seq=[]
    index=index_max
    while(index>=0):
        seq.append(str(second[index]))
        index=prev[index]
seq.reverse()
print(' '.join(seq))
    

#53D
num=int(input())
origin=list(input().split())
array=list(input().split())
positions={}
pre,last=[],[]
result=0
for i in range(num):
    if origin[i]==array[i]:
        continue
    else:
        index=len(pre)
        for j in range(i+1,num):
            pre.insert(index,j)
            last.insert(index,j+1)
            if array[j]==origin[i]:
                break
            array[i+1:j+1]=array[i:j]#this will also increase the iterator i
            #worst case n-1 swaps in n iterations
        result+=j-i#n*(n-1)<=10^6 at n=300
print (result)
for i in range(len(pre)):
    print(str(pre[i])+" "+str(last[i]))

#61A
la,lb,lout=[],[],[]
a=input()
b=input()
n=len(a)
for i in range(n):
    la.append(a[i])
    lb.append(b[i])
for i in range(len(la)):
    if(la[i]==lb[i]):
        lout.append(int(0))
    else:
        lout.append(int(1))
for i in range(n):
    print(lout[i],end="")
#58A
GirlRightFingers,GirlLeftFingers=map(int,input().split())#no pair of girls fingers touch each other
BoyRight,BoyLeft=map(int,input().split())#GBGBGBGBGBGBG(if B<G) or worst case(BBGBBGBB) 
print('YES' if 2*(GirlRightFingers+1)>=BoyLeft>=GirlRightFingers-1 or 2*(GirlLeftFingers+1)>=BoyRight>=GirlLeftFingers-1 else 'NO')

#15B
n,m,x1,y1,x2,y2=map(int,input().split())#inclusion-exclusion principle
print(n*m-2*(n-abs(x1-x2))*(m-abs(y1-y2))+min(0,n-2*(n-abs(x1-x2)))*min(0,m-2*(m-abs(y1-y2))))
print(min(0,n-2*(n-abs(x1-x2)))*min(0,m-2*(m-abs(y1-y2))))
#12E(MagicMatrix)
n = int(input())
mat=[[0]*n for r in range(n)]
l=[]
flag=[]
for i in range(n-1):
    for j in range(n-1):
        mat[i][j]=(i+j)%(n-1)+1

for i in range(n-1):
    mat[n-1][i]=mat[i][n-1]=mat[i][i]
    mat[i][i]=0

for i in range(n):
    for j in range(n):
        print(mat[i][j],end=" ")
    print()
#45D
n = int(input())
freq = [0] * 10000001
ans = [0] * 100
a = []
for i in range(n):
    l , r = map(int, input().split())
    temp = (r, l, i)
    a.append(temp)
a.sort()
print(a)
for i in range(n):
    r,l,v=a[i][0],a[i][1],a[i][2]
    for j in range(l,r+1):
        if freq[j]==0:
            freq[j]=1
            ans[v]=j
            break
for i in range(n):
    print(ans[i]," ",end="")
#45F(Code for goats and wolves(greedy))
def P(x):
    print(x)
    exit(0)
m,n=map(int, input().split())#m goats and wolves initially,n is boats capacity
flag=0
ans=0
if(n==1):P(-1)
if(m==3 and n==2 ):P(11)
if(m==5 and n==3 ):P(11)
while(True):
    if(m+m<=n):P(ans+1)
    if(m<=n):
        if(m==n):P(ans+5)
        else:P(ans+3)
    if(not flag):
         m=m-(n-2)
         ans=4
         flag=1
    else:
        if((n//2)==1):
            P(-1)
        m=m-((n//2)-1)
        ans+=2
    
#45I
n = int(input())
values = list(map(int, input().split()))
values.sort()
CountNeg=0
for i in values:
    if i<0:CountNeg+=1
    elif i>=0:break
CountPos=n-CountNeg
if(CountNeg%2==0):
    print(values)
    System.exit()
elif(CountNeg%2==1):
    for i in range(0,CountNeg-1):
        print(values[i],end=" ")
    for i in range(CountNeg-1+1,n):
        print(values[i],end=" ")
#33C
n = int(input())
values = list(map(int, input().split()))
best_infix=infix=0#theory of intersection of sets S1,S2,S3 =>S=S1+S2+S3,S-S1=S2+S3
for x in values:#we have to maximize S1=>subsequence that has the biggest sum
    infix=max(0,infix+x)
    best_infix=max(best_infix,infix)
print(2*best_infix-sum(values))#-(S-S1)+(S1)=(2*S1-S)
#34B
N, M = map(int, input().split())
A = list(map(int, input().split()))
A.sort()#due to this negative nos. gets added and positive nos. gets reduced upto m
ans=0
for i in range(M):
    if A[i]<0:
        ans-=A[i]
print(ans)
        
#39B
n=int(input())
a=list(map(int,input().split()))
years=[]
t=1
for growth in range(n):
    if(t==a[growth]):
        years.append(2001+growth)
        t+=1
print(t-1)
for _ in range(len(years)):
    print(years[_],end=" ")

#43C
n=int(input())
v=[0]*3
for i in(map(int,input().split())):
    v[i%3]+=1
print(v[0]//2+min(v[1],v[2]))#those tickets exactly divisible by 3 is matched with themselves,others having remainder 1 is matched with 2
#33A
n,m,k=map(int,input().split())
v=[1000000]*m
for _ in range(n):
    r,c=map(int,input().split())
    v[r-1]=min(v[r-1],c)
print(min(sum(v),k))
#24c
N,J=map(int,input().split())
Ax=[]#A[i-1]modN=(M[i]+M[i-1])/2=>M[i]=2*A[i-1]modN-M[i-1]
Ay=[]
Mx,My=map(int,input().split())
for i in range(N):
    x,y=map(int,input().split())
    Ax.append(x)
    Ay.append(y)
J%=2*N
for i in range(J):
    Mx=2*Ax[i%N]-Mx
    My=2*Ay[i%N]-My
print(Mx,My)
#42A
n,VolumeOfPan=map(int,input().split())
a,b=list(map(int,input().split())),list(map(int,input().split()))
x=b[0]/a[0]
for i in range(1,n):
    x=min(x,b[i]/a[i])#a1*x=b1=>x=b1/a1 and we want our x to minimum to maximize ingredients a to cook as much as soup possible
ans=0
for i in a:
    ans+=i*x#according to question there should be a1*x,a2*x...an*x litres
print(min(ans,V))
#18D
memory=[-1]*2002
memory[0]=0
for i in range(int(input())):
    string,n=input().split()
    n=int(n)+1
    if(string[0]=='w'):memory[n]=memory[0]
    elif(memory[n]>=0):memory[0]=max(memory[0],memory[n]+(1<<(n-1)))
print(memory[0])    
        
        
#10C#d(x)=(x-1)%9+1
def solve():
    n=int(input())
    #count[i] is the number of x's in [1,n] such that d(x)=i
    count=[0]*10
    for i in range(1+((n-1)%9+1)):
        count[i]=(n-1+9)//9
    for i in range(1+((n-1)%9+1),10):
        count[i]=n//9
    result=0
    # Count all triples (i, j, k) such that d(d(i)*d(j)) = d(k)
    for i in range(1,10):
        for j in range(1,10):
            result+=count[i]*count[j]*count[(i*j-1)%9+1]
    # For each j, there are n/j triples (i,j,k) such that i*j = k,
    # i.e., the correct cases
    for j in range(1,n+1):
        result-=n//j
    return result
print(solve())
#25B
n=int(input())
s=input()
t=s[:2+n%2]
for i in range(2+n%2,n,2):
    t+='-'+s[i:i+2]
print(t)
#29C
from collections import *
oxfordDictionary=defaultdict(int)
c=[]
for i in range(int(input())):
    x,y=map(int,input().split())
    c+=[x,y]
    oxfordDictionary[x]+=y
    oxfordDictionary[y]+=x
a,b=[q for q,k in Counter(c).items() if k==1]
q=0
while(a!=b):
    print(a)
    #a,b=b1,a1 means swapping tmp=b1,b=a1,a=tmp
    tmp=a
    a=oxfordDictionary[a]-q
    q=tmp
print(a)
#26D
n,m,k=map(int,input().split())
ansProbability=1
for i in range(1+k):
    ansProbability*=m-i#ans=m!n!/(m-(k-1))!(n+k+1)!=(m+n)C(m-k-1)/(m+n)C(m)
    ansProbability=ansProbability/(n+1+i)#ans=m*(m-1)*(m-2)...(m-k)/(n+1)*(n+2)...*(n+1+k)
if(ansProbability<0):ansProbability=0
if(ansProbability>1):ansProbability=1
print(1-ansProbability)
#27B
N=int(input())
won=[0]*100
b=[0]*100
j=0
for i in range(1,N*(N-1)//2):
    player1,player2=map(int,input().split())
    won[player1]+=1
    b[player1]+=1
    b[player2]+=1
for player in range(1,1+N):
    if(b[player]==N-2 and j==0):
        k=player
        j+=1
    elif(b[player]==N-2 and j==1):
        l=player
        break
if(won[k]>=won[l]):print(k,l)
else:print(l,k)
#44A
n=int(input())
count=0
stringList=[]
for i in range(n):
    stringList.append(input())
print(stringList)
for i in range(n):
    m=0
    for j in range(i):
        if(stringList[j]==stringList[i]):
            m=1
            break
    if(m==0):
        count+=1
print(count)
#29A
n=int(input())
s=set()
for _ in range(n):
    x,d=map(int,input().split())
    if ((x+d,-d) in s):
        print("YES")
    s.add((x,d))
print("NO")
#27A
test=[False]*3001
n=int(input())
defaultValues=list(map(int,input().split()))
for i in range(n):
    x=defaultValues[i]
    test[x]=True
for i in range(1,3000):
    if(test[i]==False):
        print(i)
        exit(0)
    
#8A
s=input()
s1=input()
s2=input()
forward=False
backward=False
temp=s.find(s1)
if(~temp):
    if~(s[temp+len(s1):].find(s2)):
        forward=True
s=s[::-1]
temp=s.find(s1)
if(~temp):
    if~(s[temp+len(s1):].find(s2)):
        backward=True
if(forward and backward):
    print("both")
elif(forward):
    print("forward")
elif(backard):
    print("backward")
else:
    print("fantasy")
#26B
brackets=input()
opening=0
count=0
for _ in range(len(brackets)):
    if brackets[_]=='(':
        opening+=1
    if brackets[_]==')' and opening>0:
        opening-=1
        count+=2
print(count)
#895A
n=int(input())
a=list(map(int,input().split()))
print(2*min(abs(180-sum(a[l:r]))for l in range(n) for r in range(l,n)))
#9E
from itertools import *
read=lambda:map(int,input().split())
def check_deg():
    for i in range(n):
        if(deg[i]>2):
            return False
    return True

def check_cycle():
    def cycle(u):
        mk.add(u)
        for v in range(n):
            if g[u][v]:
                g[u][v]-=1
                g[v][u]-=1
                if v in mk or cycle(v):
                    return True
        return False
    mk=set()
    cycle_num=0
    for i in range(n):
        if i in mk:
            continue
        if cycle(i):
            cycle_num+=1
    if cycle_num==1 and deg.count(2)==n:#doubt
        return True
    return cycle_num==0#if there are no cycles apart from above special case then we can add edges

def root(u):
    global f
    r=u
    while f[r]!=r:
        r=f[r]
    while f[u]!=r:#doubt
        f[u]=r
        u=f[u]
    return r

n,m=read()
g=[[0]*n for i in range(n)]
deg=[0]*n
f=[0]*n
for i in range(n):
    f[i]=i
for i in range(m):
    u,v=read()
    u,v=u-1,v-1
    deg[u]+=1
    deg[v]+=1
    g[u][v]+=1
    g[v][u]+=1
    f[root(u)]=root(v)
if m>n or not check_deg() or not check_cycle():
    print('NO')
else:
    print('YES')
    print(n-m)
    if n==1 and n-m>0:
        print('1 1')
        exit(0)
    for i in range(n-m):
        for u,v in combinations(range(n),2):#doubt here but i think it is taking any 2 numbers in range(n)
            #using combinations to get lexicographically minimal pair of sets
            #B2 there are no cycles so u,v are different connected components
            if deg[u]<2 and deg[v]<2 and root(u)!=root(v):
                print(u+1,v+1)
                deg[u]+=1
                deg[v]+=1
                f[root(u)]=root(v)
                break
    #if there are 2 connected components,then they could be connected by an edge without breaking
    #B1 m<n and B3 deg<2
    #Here obtained graph is just a walk and can be connected by its end point to form a funny ring
    for u,v in combinations(range(n),2):
            if deg[u]<2 and deg[v]<2:
                print(u+1,v+1)
                deg[u]+=1
                deg[v]+=1
                
#896A
s="What are you doing at the end of the world? Are you busy? Will you save us?"
s1='What are you doing while sending "'
s2='"? Are you busy? Will you send "'#f[i]=s1+f[i-1]+s2+f[i-1]+s3,f[0]=s
s3='"?"'
print(len(s),len(s1),len(s2),len(s3))#=>75,34,32,3
l1,l2,l3=len(s1),len(s2),len(s3)
def count(n):
    if n>=60:#length of f(n)>=(2*length of f(n-1))=>length of f(60)>=kmax
        return 10**20#n=0=>75
    return (1<<n)*75+((1<<n)-1)*68#2^n*75+(2^n-1)*68 #a=5,a<<3=8*a
def find(n,k):
    if(k>count(n)):return '.'#for(n=1,k=194)=>count(n)=2*75+34=150+68=218
    if n==0:return s[k-1]
    if k<=l1:return s1[k-1]
    c=count(n-1)
    k=k-l1
    if k<=c:
        return find(n-1,k)
    k=k-c
    if k<=l2:
        return s2[k-1]
    k=k-l2
    if k<=c:
        return find(n-1,k)
    k=k-c
    if k<=l3:
        return s3[k-1]

q=int(input())
ans=''
while q:
    n,k=map(int,input().split())
    while(n>70 and k>34):
        k=k-34
        n=n-1
    if(n>0 and k<=34):
        ans+=s1[k-1]
    else:
        ans+=find(n,k)
    q=q-1
print(ans)

    
#867A#Between the Offices
n=int(input())
string=input()
if string[0]=='S' and string[-1]=='F':
    print("YES")
#35D
#Greedy food eating
num_days,food=list(map(int,input().split()))
eats=list(map(int,input().split()))
values=[]
for i in range(num_days):
    values.append(eats[i]*(num_days-i))#assume we are making it eat on ith day
values.sort()
for value in values:
    if value> food:
        break
    count+=1
    food=food-value
print(count)
#43A
n=int(input())#0R print(sorted([input()for i in range(n)])[int(n/2)])
TeamList=[]
for _ in range(n):
    TeamList.append(input())
count=0
for i in range(n):
    if TeamList[0]!=TeamList[i]:
        count+=1
        p=i
count2=n-count
if(count<count2):print(TeamList[0])
else:print(TeamList[p])

#43B
heading=input()
text=input()
print('NO' if any(heading.count(letter)<text.count(letter) for letter in text if letter!=' ')else 'YES')
#15A
N,T=map(int,input().split())
X=[0]*N
A=[0]*N
for i in range(N):
    X[i],A[i]=map(int,input().split())
x=sorted(X)
a=sorted(A)
print(x)
print(a)
ans=2#to close 2 extremes
for i in range(N-1):
    distance=x[i+1]-x[i]+a[i]/2+a[i+1]/2
    if T<distance:
        ans+=2
    elif T==distance:
        ans+=1
print(ans)
    
#6C
n=int(input())
A=list(map(int,input().split()))
i=0
a,b=0,0
x,y=0,0
while i<n:
    if x<=y:
        x+=A[i]
        i+=1
        a+=1
    else:
        y+=A[n-1]
        n-=1
        b+=1
print(a,b)

#18B
n,d,m,L=map(int,input().split())
z=(n-1)*m+L
for i in range(d,(m+1)*d,d):
    if i>z or i%m>L:#what happens is if i>(k-1)m+l or i%m>l Bob will not be on the edge
        print(i)#falls down
        exit()
print(z-z%d+d)#otherwise last k=n
#19B
N=int(input())
f=[0]*(2001)
f[0]=0
for _ in range(1,1+N):
    f[_]=10**9
for i in range(N):
    #01 KnapSack Problem where a=val,b=weight
    a,b=map(int,input().split())
    a=a+1
    for j in range(N,0,-1):
        if f[max(j-a,0)]+b<f[j]:
            f[j]=f[max(j-a,0)]+b
print(f[N])
#23A
s=input()
for i in range(len(s),0,-1):
    for j in range(len(s)-i+1):
        if s[j:j+i] in s[j+1:]:
            print(i)
            exit()
#23C
T=int(input())
for k in range(T):
    n=int(input())
    boxes=list()
    for i in range(2*n-1):
        boxes.append(tuple(map(int,input().split()))+(i+1,))
    boxes=sorted(boxes)[::-1]
    print("YES")
    print(boxes[0][2],)
    for i in range(1,2*n-1,2):
        if(boxes[i][1]<boxes[i+1][1]):
            print(" ",boxes[i+1][2],)
        else:
            print(" ",boxes[i][2],)
    print("")
#2A
p=[]
r={}
n=int(input())
for i in range(n):
    a,b=input().split()
    b=int(b)
    r[a]=r.get(a,0)+b
    p.append([r[a],a])
    m=max(r.values())
for n,a in p:
    if n>=m and r[a]>=m:
        print(a)
        break
#2B
n=int(input())
a=[map(int,input().split()) for i in range(n)]
def f(m,k):
    r=0
    while m and m%k==0:
        m=m//k
        r+=1
    return r
def dp(k):
    b=[[f(x,k) for x in c]for c in a]
    for i in range(1,n):
        b[i][0]+=b[i-1][0]
        b[0][i]+=b[0][i-1]
    for i in range(1,n):
        for j in range(1,n):
            b[i][j]+=min(b[i][j-1],b[i-1][j])
    ans=''
    i,j=n-1,n-1
    while i+j:
        if i==0 or (i*j and b[i][j-1]<b[i-1][j]):
            j=-1
            ans+='R'
        else:
            i=-1
            ans+='D'
    return(b[n-1][n-1],ans[::-1])
(o,s)=min(dp(2),dp(5))
for i in range(n):
    if o>1:
        for j in range(n):
            if a[i][j]==0:
                o=1
                s='R'*j+'D'*(n-1)+'R'*(n-1-j)
print(o,'\n',s)
#13A
def findDigitSum(num,base):
    total=0
    while num>0:
        total+=num%base
        num=num//base
    return total
A=int(input())
totalSum=0
for k in range(2,A):
    totalSum+=findDigitSum(A,k)
currentGcd=gcd(totalSum,A-2)
print(str(totalSum//currentGcd)+"/"+str((A-2)//currentGcd))#important

#1A
def TheatreSquare(n,m,a):
    return((-n//a)*(-m//a))#-sign is important
x=list(map(int,input().split()))
print(TheatreSquare(x[0],x[1],x[2]))
