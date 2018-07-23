def sumDigits(n):
    return sum(map(int,list(str(n))))
n=int(input())
best=1
best_sum=1
for k in range(1,n+1):
    if n%k==0:
        the_sum=sumDigits(k)
        if the_sum>best_sum:
            best_sum=the_sum
            best=k
print(best)




#Factorial Array
from bisect import bisect_left
MOD=10**9
f=[None]*40
f[1]=s=1

for i in range(2,40):
    s=s*i
    f[i]=s%MOD
R=lambda:map(int,input().split())
n,m=R()

a=R()
v=[[] for _ in range(40)]
for i,x in enumerate(a):
    if x<40:
        v[x].append(i)

for _ in range(m):
    t,l,r=R()
    l=l-1
    if t==1:
        il,ir=(bisect_left(v[39],x) for x in (l,r))
        for j in range(38,-1,-1):
            il1,ir1=(bisect_left(v[j],x) for x in (l,r))
            v[j+1][il:ir]=v[j][il1:ir1]
            il,ir=il1,ir1
            #print(v)
    elif t==2:
        s=0
        for i in range(1,40):
            s=s+f[i]*(bisect_left(v[i],r)-bisect_left(v[i],l))
            #print(s)
            s=s%MOD
    else:
        for i in range(1,40):
            j=bisect_left(v[i],l)
            if j<len(v[i]) and v[i][j]==1:
                del v[i][j]
                break
        if r<40:
            v[r].insert(bisect_left(v[r],l),l)
            

#Dynamic Array
def get_seq_index(x,last_ans,n):
    return (x^last_ans)%n

def get_index(y,size):
    return y%size

def main():
    n,q=map(int,input().split())
    last_ans=0
    sequences=[[] for _ in range(n)]
    for i in range(q):
        comm,x,y=map(int,input().split())
        seq_index=get_seq_index(x,last_ans,n)
        if comm==1:
            sequences[seq_index].append(y)
        else:
            seq_len=len(sequences[seq_index])
            item_index=get_index(y,seq_len)
            item=sequences[seq_index][item_index]
            last_ans=item
            print(item)

if __name__=="__main__":
    main()
