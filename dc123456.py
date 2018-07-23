    
compute_207()
def compute_206():
    #for x=1_2_3_4_5_6_7_8_9_0 to be a perfect square 2nd last digit=0 then
    #last 3 digits are 900 so either 30 or 70 comes into mind for last 2 digit of x
    #so we will increase the step by lower value 30 
    s=(int(1020304050607080900**0.5)//100)*100+30
    l=(int(1929394959697989900)**0.5)
    x=s
    while x<l:
        if str(x**2)[::2]=="1234567890":
            print(x)
            return x**2
            break
        if x%100==30:
            #since 70 is next possibility so increase by 40
            x+=40
        else:
            #else increase by 2*30
            x+=30
print(compute_206())


def compute(n):
    divisors = [1] * (n + 1)  # Invalid for indexes 0 and 1
    for i in range(2, (len(divisors) + 1) // 2):
        for j in range(i * 2, len(divisors), i):# example take 3 then all multiples of 3 less than 10**7 will contain 3 like 6,9,12,18..so their count is incrementated by 1
            divisors[j] += 1
    return divisors
print(compute(12))


def compute_40():
    s="".join(str(i) for i in range(1,1000000))
    ans=1
    for i in range(1,7):
        ans=ans*int(s[10**i-1])
    return str(ans)
print(compute_40())




def digits(n):
    cnt=0
    while n!=0:
        cnt+=1
        n=n//10
    return cnt
print(digits(123456788))


string=input()
l=string.split()
d={}
for word in l:
    if(word[0] not in d.keys()):
        d[word[0]]=[]
        d[word[0]].append(word)
    else:
        if(word not in d[word[0]]):
            d[word[0]].append(word)
for k,v in d.items():
    print(k,":",v)
    

keys=[]
values=[]
d=dict(zip(keys,values))

n=int(input())
d={x:x**2 for x in range(1,n+1)}
key=int(input())
if key in d:
    del d[key]
print(d)

key=int(input())
value=int(input())
d={}
d.update({key:value})
print(d)



d={'A':1,'B':2,'C':3}
print(sum(d.values()))
key=input()
if key in d.keys():
      print(d[key])
else:
      print("Key isn't present!")
