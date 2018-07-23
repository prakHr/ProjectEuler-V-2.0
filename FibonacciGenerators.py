target=pow(10,17)
fib=[1]
zeck=[0,1,2,3]
x_n=1
x_np=1
i=0
while x_n <= target:
  x_n,x_np = x_np,x_n+x_np
  fib.append(x_n)
  if i>=4:
      #Properties are:
      #Given f(x) is the Fibonacci function:
#sum(1, f(x)) = sum(1, f(x-1)) + sum(1, f(x-2)) + f(x-2) - 1

#If f(x-1) < a < f(x):
#sum(1, a) = sum(1, f(x-1)) + sum(1, a - f(x-1)) + (a - f(x-1))
      zeck.append(zeck[i-1]+zeck[i-2]+fib[i-2]-1)
  i+=1

def res(n):
  if n==0:
      return 0
  if n==1:
      return 1
  for i in range(0, len(fib)):
      if fib[i]>n:
          break
  i-=1
  #Thm. Every positive integer has exactly one Zeckendorf representation.
  #n-fib[i],where fib[i] is largest term in fibonacci series<=n
  #So,sum of zeckendorf representation is found recursively zeck[i]+res(n-fib[i])
  #Suppose largest fib[a]<n such that n is a representation
  #n=fib[a]+fib[b1]+...fib[bk]
  #then n-fib[a] is another representation of same property so add it
  return zeck[i]+res(n-fib[i])+(n-fib[i])
start=[[0,-1],[0,1],[-3,-2],[-3,2],[-4,-5],[-4,5],[2,-7],[2,7]]
#print(start[0][1])
#A(x)=xG1+x^2G2+...
#A(x)=sigma(x^k*Gk)from k=0 to inf
#A(x)=x+4x^2+sigma(x^k*(Gk-1+Gk-2))from k=3 to inf
#Then solve using Quadratic Solver
nuggets=set()
sum=0
for j in range(0,len(start)):
    k,b=start[j][0],start[j][1]
    for i in range(0,30):
        new_k=-9*k-4*b-14
        new_b=-20*k-9*b-28

        k=new_k
        b=new_b
        if k>0 :
            nuggets.add(k)

list_nuggets=sorted(list(nuggets))
for i in range(0,30):
    sum+=list_nuggets[i]
print(sum)

import itertools



def Solve_186():
    #def InitNetwork():
    N=1000000
    parent=[i for i in range(0,N)]
    rank=[0]*(N)
    group_size=[1]*N
       
       
    def FindSet(x):
        if x!=parent[x]:
            parent[x]=FindSet(parent[x])
        return parent[x]
    def GetGroupSize(x):
        return group_size[FindSet(x)]
    def Link(x,y):
        if x==y:
            return
        if rank[x]>rank[y]:
            parent[y]=x
            group_size[x]+=group_size[y]
            group_size[y]=0
        else:
            parent[x]=y
            group_size[y]+=group_size[x]
            group_size[x]=0
            if rank[x]==rank[y]:
                rank[y]+=1
    def Union(x,y):
        Link(FindSet(x),FindSet(y))

    #InitNetwork()
    S=[0]*55
    dial_count=0
    is_caller=True
    for k in range(1,55+1):
        S[k-1]=(100003-200003*k+300007*k*k*k)%1000000
        if S[k-1]<0:
            S[k-1]+=1000000
        if is_caller:
            caller_number=S[k-1]
        elif (caller_number!=S[k-1]):
            Union(caller_number,S[k-1])
            dial_count+=1
        is_caller=not is_caller
    k0=0
    for k in itertools.count(56):
        k0=(k-1)%55
        S[k0]=(S[(k-25)%55]+S[(k-56)%55])%1000000
        if is_caller:
            caller_number=S[k0]
        elif (caller_number!=S[k0]):
            Union(caller_number,S[k0])
            dial_count+=1
        is_caller=not is_caller
        if(GetGroupSize(524287)>=990000):
            print(dial_count)
            break
Solve_186()
        
