#syllableForNicks=['N','IHo','K','S']
#entries=nltk.corpus.cmudict.entries()
#[word for word,pron in entries if pron[-4:]==syllableForNicks]

#puzzleLetters=nltk.FreqDist('egivrvonl')
#always='r'
#wordlist=nltk.corpus.words.words()
#[w for w in wordlist if len(w)>=4+1(r)+1(v appear 2X) and always in w and nltk.FreqDist(w)<=puzzleLetters]

#from nltk.corpus import brown
#brown.categories()
#t=brown.words(categories='news')
#brown.sents(categories=['news','romance'])
#modals=['can','might']
#fdist=nltk.FreqDist([w.lower() for w in t])
#for m in modals:print(m+':',fdist[m])

#cfd=nltk.ConditionalFreqDist((genre,word) for genre in brown.categories() for word in brown.words(categories=genre) )
#cfd.tabulate(conditions=genres,samples=modals)

#from nlyk import udhr
#languages = ['Chickasaw', 'English', 'German_Deutsch',... 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
#cfd=nltk.ConditionalFreqDist((languages,len(word) for l in languages for word in udhr.words(l+'-Latin1')))
#cfd.plot(cumulative=True)

#from nltk.book import *(contains text1)
#text1.concordance("monstrous")=>permits us to see the word in contexts
#text1.similar("monstrous")=>what other words in similar range of contexts
#text2.common_contexts(["monstrous","very"])
#text4.dispersion_plot()
#text3.generate()
#sorted(set(text3))
#text5.count('lol')
#define lexicalDiversity(text):return len(text)/len(set(text))
#define percent(text,word):100*text.count(word)/len(text)
#sent1=['Monty','Python'],sent2=['And','the','Holy','-','Grail','.']
#sent1+sent2=?
#name='ShinChan'
#name*2=?
#' '.join(sent1)=?(a single string)
#'a single string'.split()=a list of words

#fdist=FreqDist([len(w) for w in text1])
#So if fdist.keys() is between[1,20] means there are 20 different word lengths
#fdist.max() will give us length of the most freqent words 
#fdist.items() will give us (k,v) pairs according to sorted v's
#fdist[3] will give us the freq-value of len(w)=3 
#fdist.freq(3) will give us the same in percent

#len(set(text1)) and len(set([w.lower() for w in text1])) are not same!

#Ace=FreqDist(LuffySTale)
#vocub1=Ace.keys()
#Ace['Sabo']
#Ace.plot(32,cumulative=True)=>32 most frequent words Ace spoke if he didnt mentioned sabo
#rarityCasesOfWords=Ace.hapaxes()

#fdist5=FreqDist(text5)
#LongWords=[w for w in set(text5) if len(w)>12 and fdist5[w]>7]

#text4.collocations()




#from matplotlib import pyplot as plt
#plt.hist([1,4,4,5,5,5,9])
#plt.show()
#So words=word_tokenize("thsre wewr werw w!")
#wordLengths=[len(w) for w in words]

#from nltk.tokenize import regexp_tokenize
#from nltk.tokenize import TweetTokenizer
#pattern1 = r"#\w+"
#regexp_tokenize(tweets[0], pattern1)
#tknzr = TweetTokenizer()
#all_tokens = [tknzr.tokenize(t) for t in tweets]
#print(all_tokens)

#from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
#sentences=sent_tokenize(scene1)
#FourthSentenceTokenized=word_tokenize(sentences[3])
#UniqueTokens=set(word_tokenize(scene1))

import re
matchDigitsAndWords=('(\d+|\w+)')#examples for ranges and groups [A-Za-z]+,[0-9],[A-Za-z\-\.]+,
#matches a,- and z(a-z),matches spaces or comma(\s+l,)
print(re.findall(matchDigitsAndWords,'fegewg dwa 11 ferfe'))

myStr="Let's write RegEx!"
print(re.findall(r"\w+",myStr))
print(re.findall(r"\w",myStr))
print(re.findall(r"\s+",myStr))
print(re.findall(r"[a-z]",myStr))

print(re.split('\s+','Split on spaces.'))
#to match digits use \d
#to match space use \s to find them and seperate them from a string
#to match any letter or symbol use .*(wildcard character)
#with the help of + or * we can grab repeats of single letters
#for no_spaces use \S
#for lowercase letters [a-z]

#split(string on regex),findall(patterns),search(for a pattern),match(an string or substring based on pattern)
word_regex='\w+'#takes first word

#always pattern first and string second
#return an iterator,string or match object
print(re.match(word_regex,'hi there!'))

print(re.match('abc','abcdef'))

import re
my_string='hi there!'
# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings,my_string ))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))

