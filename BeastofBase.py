def baseConverter(n):
    modulo=7#this is a pattern on pascal's triangle for number not divisible by modulo
    res=1
    while n>0:
        res=res*(n%modulo+1)
        n=n//modulo
    return res

def compute_148():
    Limit=10**9
    mod=7
    count=1
    base7=[1]*12
    for row in range(1,Limit):
        base7[0]+=1
        carrypos=0
        while base7[carrypos]==mod+1:
            base7[carrypos]=1
            base7[carrypos+1]+=1
            carrypos+=1
        found=1
        print(base7)
        for x in base7:
            if x!=1:
                found=found*x
        count+=found
    return str(count)
        
        
#x=sum(baseConverter(n) for n in range(0,10**9))
print(compute_148())
