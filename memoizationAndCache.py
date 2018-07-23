def AckermannFunction():#Not working
    
    y_range=30000000
    x_range=6
    result=0
    cache = [[0 for x in range(x_range)] for y in range(y_range)] 
    def ACK(m,n):
        if m>=x_range or n>=y_range:
            return 0
        if cache[m][n]:
            return cache[m][n]
        if m==0:
            cache[m][n]=n+1
            return n+1
        elif n==0:
            result=ACK(m-1,1)
            cache[m][n]=result
            return result
        else:
            result=ACK(m,n-1)
            result=ACK(m-1,result)
            cache[m][n]=result
            return result
    print(ACK(3,4))

AckermannFunction()
        


def has_divider(d):
    remaining_hash=set()
    T=(1,1,1)
    while 1:
        T=(T[1],T[2],(T[0]+T[1]+T[2])%d)
        if T[2]==0:
            return True
        #h(x,y,z)=d^2*x+d*y+*z
        value=T[0]*d*d+T[1]*d+T[2]
        if value in remaining_hash:
            return False
        else:
            remaining_hash.add(value)
            
count=0
for d in range(1,10000,2):
    if not has_divider(d):
        count+=1
        if count is 124:
            print(d)
            break




S={}
def f(x):
    if x==0:
        return 1
    if x in S:
        return S[x]
    if x%2==1:
        return f((x-1)//2)
    S[x]=f(x//2)+f(x//2-1)
    return S[x]
print(f(10**25))


#180 is fermatlastThmf(1,n)+(xy+yz+zx)f(1,n-2)-(xyz)f(1,n-3)=f(1,n-1)(x+y+z)
