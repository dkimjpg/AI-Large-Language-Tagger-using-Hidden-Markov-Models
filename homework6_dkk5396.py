############################################################
# CMPSC 442: Hidden Markov Models
############################################################

student_name = "David Kim"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

#import collections
from collections import defaultdict

############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    #openMessage = open(path) #, encoding="utf8")
    corpusList = []
    #The code below was previously used and worked successfully during testing, but when uploading this 
    # file to Gradescope, Gradescope's autograder would continuously fail to execute correctly. This code
    # still works, but since it is unusable on Gradescope, it was replaced with an alternative implementation.
    # I think the problem may be with openMessage.readlines(), Gradescope may not be able to properly run that code.
    """
    for line in openMessage.readlines():
        currentList = []
        currentLine = line.split() #split by whitespace, since that's what each entry is separated by
        for part in currentLine:
            partList = part.split("=")
            currentList.append(partList)
        corpusList.append(currentList)
    """
    
    #The following code to open the file actually works, so this will be used.
    with open(path) as file:
        for line in file:
            currentList = []
            for part in line.split(): #split by whitespace, since that's what each entry is separated by
                partList = part.split("=")
                currentList.append(tuple(partList))
            corpusList.append(currentList)
    
    return(corpusList)
    #pass

#def getTokens():

def extractTag(line, position):
    tag = line[position]#.split("=")
    #print(tag)
    tagExtraction = tag[1]
    return tagExtraction

def extractToken(line, position):
    token = line[position]#.split("=")
    tokenExtraction = token[0]
    return tokenExtraction

TAGS = ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X')

class Tagger(object):
    
    def __init__(self, sentences):
        smoothingProb = 1e-10
        self.pi = {}
        self.alpha = {}
        self.beta = {}
        
        #filling all the dictionaries with 0's
        for tag in TAGS:
            self.pi[tag] = 0
            self.alpha[tag] = {}
            for innerTag in TAGS:
                self.alpha[tag][innerTag] = 0
            self.beta[tag] = defaultdict(int) #beta needs a default dict since each tag in beta will be a dictionary with all keys initially set to 0

        #begin counting for pi, alpha, and beta
        for line in sentences:
            #pi counting
            getFirstTag = extractTag(line, 0)
            self.pi[getFirstTag] += 1

            #beta counting for first element in line (this is necessary because the for loop after this skips the first element)
            getFirstToken = extractToken(line, 0)
            self.beta[getFirstTag][getFirstToken] += 1

            #alpha counting and beta counting

            #Extra Note: I might have to use Counter() for alpha and beta counting if my implementation is not fast enough. 
            for currentToken in range(1, len(line)):
                #alpha counting
                getTag = extractTag(line, currentToken)
                getPreviousTag = extractTag(line, currentToken - 1)
                self.alpha[getPreviousTag][getTag] += 1

                #beta counting
                getToken = extractToken(line, currentToken)
                self.beta[getTag][getToken] += 1
                            
        #counting for pi, alpha, and beta end here.


        #Begin the smoothing algorithms
        #pi smoothing
        #finds total for pi based on pi counting
        piTotal = sum(self.pi.values()) + (smoothingProb * len(self.pi.keys())) #may or may not need to add 1 to the len(self.pi,keys()) since this is Laplace smoothing
        
        #applies smoothing
        for tag in TAGS:
            self.pi[tag] = float(float(self.pi[tag] + smoothingProb) / piTotal)
        

        #alpha smoothing
        for tag in TAGS:
            #finds total for alpha based on alpha counting
            alphaTotal = 0
            if self.alpha[tag]: #checks if the value for self.alpha[tag] is not empty, or in other words, looks like {}
                alphaTotal = alphaTotal + sum(self.alpha[tag].values()) + (smoothingProb * len(self.alpha[tag].keys())) #assuming that each dictionary entry has a dictionary within it
            
            #applies smoothing
            for innerTag in TAGS:
                self.alpha[tag][innerTag] = float(float(self.alpha[tag][innerTag] + smoothingProb) / alphaTotal)
        

        #beta smoothing
        for tag in TAGS:
            #finds total for beta based on beta counting
            betaTotal = smoothingProb
            if self.beta[tag]:
                betaTotal = sum(self.beta[tag].values())
            #applies smoothing
            for token in self.beta[tag]:
                self.beta[tag][token] = float(float(self.beta[tag][token] + smoothingProb) / betaTotal)
            self.beta[tag]["<UNK>"] = smoothingProb / betaTotal

        #pass

    def most_probable_tags(self, tokens):        
        probableList = []

        #loop through all the tokens, and for each token, check every single tag to get the most probable 
        #tag for that token. Then append it into the probableList.
        for token in tokens:
            probableVal = -1
            currentVal = 0
            probableTag = ''
            for tag in TAGS:                
                if token in self.beta[tag]: #check if the token is in the beta probabilities, if not, set it as a <UNK> since it's unknown
                    currentVal = self.beta[tag][token]
                else:
                    currentVal = self.beta[tag]["<UNK>"]
                
                if currentVal > probableVal: #compare the current value with the chosen probable value, and if the current value is larger, make it the new probable value
                    probableVal = currentVal
                    probableTag = tag
            probableList.append(probableTag)
        return probableList
        #pass

    def viterbi_tags(self, tokens):
        #initialize two 2d lists with 0's and 0.0's
        #viterbi is a list that will be used in conjunction with the viterbi algorithm. It's more convenient to use a 2d list for this.
        #backPoint is the list that will be used to keep track of backpointers
        viterbi = [[0.0 for xtag in TAGS] for xtoken in tokens]
        backPoint = [[0 for xtag in TAGS] for xtoken in tokens]

        #Starting node of the HMM
        for tag in range(0, len(TAGS)):
            currentTag = TAGS[tag]
            back = 0
            if tokens[0] in self.beta[currentTag]: #check if the token is in the beta probabilities, if not, set it as a <UNK> since it's unknown
                back = self.beta[currentTag][tokens[0]]
            else:
                back = self.beta[currentTag]["<UNK>"]
            viterbi[0][tag] = self.pi[currentTag] * back
        
        #all other nodes in the HMM
        for token in range(1, len(tokens)):
            for tag in range(0, len(TAGS)):
                probableVal = -1
                probableTagIndex = 0
                for innerTag in range(0, len(TAGS)):
                    firstTag = TAGS[innerTag]
                    secondTag = TAGS[tag]
                    calcVal = viterbi[token - 1][innerTag] * self.alpha[firstTag][secondTag]

                    if calcVal > probableVal:
                        probableVal = calcVal
                        probableTagIndex = innerTag
                backPoint[token][tag] = probableTagIndex #logs the tag index that should be pointed back to
                secondTag = TAGS[tag]
                back = 0
                if tokens[token] in self.beta[secondTag]: #check if the token is in the beta probabilities, if not, set it as a <UNK> since it's unknown
                    back = self.beta[secondTag][tokens[token]]
                else:
                    back = self.beta[secondTag]["<UNK>"]
                viterbi[token][tag] = probableVal * back

        #begin setting up for backtracking
        probableTagIndex = 0
        probableVal = -1
        probableList = []

        for tag in range(0, len(TAGS)):
            if viterbi[-1][tag] > probableVal: #compares the last element's values to the currently chosen probable value and sets new probable value to the highest of the last element's values
                probableVal = viterbi[-1][tag]
                probableTagIndex = tag
        probableList.append(TAGS[probableTagIndex])
        previousPoint = probableTagIndex

        #backtracking starts here
        for point in range(len(tokens) - 2, -1, -1): #need to count down instead of up for easier backtracking
            previousPoint = backPoint[point + 1][previousPoint]
            probableList.append(TAGS[previousPoint])
        
        return list(reversed(probableList))
        #pass


#####################################################
# Test Cases
#####################################################

print("Question 1\n")
c = load_corpus("brown-corpus.txt")
print(c[1402])
print()
print(c[1799])

print("\nQuestion 3\n")
c = load_corpus("brown-corpus.txt")
t = Tagger(c)
print(t.most_probable_tags(["The", "man", "walks", "."]))
print(t.most_probable_tags(["The", "blue", "bird", "sings"]))

print("\nQuestion 4\n")
c = load_corpus("brown-corpus.txt")
t = Tagger(c)
s = "I am waiting to reply".split()
print(t.most_probable_tags(s))
print(t.viterbi_tags(s))
print()

s = "I saw the play".split()
print(t.most_probable_tags(s))
print(t.viterbi_tags(s))